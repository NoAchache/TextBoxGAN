import time
import tensorflow as tf
import tensorflow as tf

from utils.tf_utils import allow_memory_growth
from training_step import TrainingStep
from utils import LogSummary, LossTracker
from config import cfg
from dataset_utils.data_loader import load_dataset
from models.model_loader import ModelLoader


class Trainer(object):
    def __init__(self):
        if cfg.allow_memory_growth:
            allow_memory_growth()

        self.batch_size = cfg.batch_size
        self.strategy = cfg.strategy
        self.max_epochs = cfg.max_epochs
        self.n_samples = min(self.batch_size, cfg.n_samples)

        self.print_step = cfg.print_step

        self.save_step = cfg.save_step

        self.log_step = cfg.log_step
        self.log_dir = cfg.log_dir
        self.log_summary = LogSummary()

        # set optimizer params
        self.g_opt = self.update_optimizer_params(cfg.g_opt)
        self.d_opt = self.update_optimizer_params(cfg.d_opt)
        self.pl_mean = tf.Variable(
            initial_value=0.0,
            name="pl_mean",
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        self.model_loader = ModelLoader()
        # create model: model and optimizer must be created under `strategy.scope`
        (
            self.discriminator,
            self.generator,
            self.g_clone,
        ) = self.model_loader.initiate_models()

        # set optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(
            self.d_opt["learning_rate"],
            beta_1=self.d_opt["beta1"],
            beta_2=self.d_opt["beta2"],
            epsilon=self.d_opt["epsilon"],
        )
        self.g_optimizer = tf.keras.optimizers.Adam(
            self.g_opt["learning_rate"],
            beta_1=self.g_opt["beta1"],
            beta_2=self.g_opt["beta2"],
            epsilon=self.g_opt["epsilon"],
        )

        self.training_step = TrainingStep(
            self.generator,
            self.discriminator,
            self.g_optimizer,
            self.d_optimizer,
            self.g_opt["reg_interval"],
            self.d_opt["reg_interval"],
            self.batch_size,
            self.pl_mean,
        )
        # TODO:add loss tracking
        # TODO:add tf logging (image with label as name)

        # TODO: what is zis?
        self.manager = self.model_loader.load_checkpoint(
            ckpt_kwargs={
                "d_optimizer": self.d_optimizer,
                "g_optimizer": self.g_optimizer,
                "discriminator": self.discriminator,
                "generator": self.generator,
                "g_clone": self.g_clone,
                "pl_mean": self.pl_mean,
            },
            model_description="Full model",
            expect_partial=False,
            ckpt_dir=cfg.ckpt_dir,
            max_to_keep=10,
        )

    @staticmethod
    def update_optimizer_params(params):
        params_copy = params.copy()
        mb_ratio = params_copy["reg_interval"] / (params_copy["reg_interval"] + 1)
        params_copy["learning_rate"] = params_copy["learning_rate"] * mb_ratio
        params_copy["beta1"] = params_copy["beta1"] ** mb_ratio
        params_copy["beta2"] = params_copy["beta2"] ** mb_ratio
        return params_copy

    def train(self):
        with self.strategy.scope():
            dataset = load_dataset(
                shuffle=True, epochs=self.max_epochs, batch_size=self.batch_size
            )
            dataset = self.strategy.experimental_distribute_dataset(dataset)

            # wrap with tf.function
            dist_train_step = tf.function(self.training_step.dist_train_step)
            dist_gen_samples = tf.function(self.log_summary.dist_gen_samples)

            # start actual training
            print("Start Training")

            # setup tensorboards
            train_summary_writer = tf.summary.create_file_writer(self.log_dir)

            # loss metrics
            metric_d_loss = tf.keras.metrics.Mean("d_loss", dtype=tf.float32)
            metric_g_loss = tf.keras.metrics.Mean("g_loss", dtype=tf.float32)
            metric_d_gan_loss = tf.keras.metrics.Mean("d_gan_loss", dtype=tf.float32)
            metric_g_gan_loss = tf.keras.metrics.Mean("g_gan_loss", dtype=tf.float32)
            metric_r1_penalty = tf.keras.metrics.Mean("r1_penalty", dtype=tf.float32)
            metric_pl_penalty = tf.keras.metrics.Mean("pl_penalty", dtype=tf.float32)

            # start training
            zero = tf.constant(0.0, dtype=tf.float32)  # TODO: delete
            t_start = time.time()

            for real_images in dataset:
                step = self.g_optimizer.iterations.numpy()

                # g train step
                do_r1_reg = (step + 1) % self.d_opt["reg_interval"] == 0
                do_pl_reg = (step + 1) % self.g_opt["reg_interval"] == 0

                gen_losses, disc_losses = dist_train_step(
                    real_images, do_r1_reg, do_pl_reg
                )
                reg_g_loss, g_loss, pl_penalty = gen_losses
                reg_d_loss, d_loss, r1_penalty = disc_losses

                # TODO: rename all the reg_g_loss, etc...

                # update g_clone
                self.g_clone.set_as_moving_average_of(self.generator)

                # update metrics
                metric_d_loss(d_loss)
                metric_g_loss(g_loss)
                metric_d_gan_loss(d_gan_loss)
                metric_g_gan_loss(g_gan_loss)
                metric_r1_penalty(r1_penalty)
                metric_pl_penalty(pl_penalty)

                # get current step
                step = self.g_optimizer.iterations.numpy()

                # TODO: export to log summary
                # save to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar("d_loss", metric_d_loss.result(), step=step)
                    tf.summary.scalar("g_loss", metric_g_loss.result(), step=step)
                    tf.summary.scalar(
                        "d_gan_loss", metric_d_gan_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "g_gan_loss", metric_g_gan_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "r1_penalty", metric_r1_penalty.result(), step=step
                    )
                    tf.summary.scalar(
                        "pl_penalty", metric_pl_penalty.result(), step=step
                    )

                # save every self.save_step
                if step % self.save_step == 0:
                    self.manager.save(checkpoint_number=step)

                # save every self.image_summary_step
                if step % self.image_summary_step == 0:
                    # add summary image
                    test_z = tf.random.normal(
                        shape=(self.n_samples, self.g_params["z_dim"]),
                        dtype=tf.dtypes.float32,
                    )

                    summary_image = dist_gen_samples(test_z, self.g_clone)

                    # convert to tensor image
                    summary_image = self.convert_per_replica_image(
                        summary_image, self.strategy
                    )

                    with train_summary_writer.as_default():
                        tf.summary.image("images", summary_image, step=step)

                # print every self.print_steps
                if step % self.print_step == 0:
                    elapsed = time.time() - t_start
                    print(
                        self.log_template.format(
                            step,
                            elapsed,
                            d_loss.numpy(),
                            g_loss.numpy(),
                            d_gan_loss.numpy(),
                            g_gan_loss.numpy(),
                            r1_penalty.numpy(),
                            pl_penalty.numpy(),
                        )
                    )

                    # reset timer
                    t_start = time.time()

            # save last checkpoint
            step = self.g_optimizer.iterations.numpy()
            self.manager.save(checkpoint_number=step)
            return

    @staticmethod
    def convert_per_replica_image(nchw_per_replica_images, strategy):
        as_tensor = tf.concat(
            strategy.experimental_local_results(nchw_per_replica_images), axis=0
        )
        as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
        as_tensor = tf.cast(as_tensor, tf.uint8)
        return as_tensor


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
