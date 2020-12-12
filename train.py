ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth
    allow_memory_growth()


import tensorflow as tf


from config import cfg
from training_step import TrainingStep
from utils import TensorboardWriter, LossTracker
from dataset_utils.data_loader import load_dataset
from models.model_loader import ModelLoader
from aster_ocr_utils.aster_inferer import AsterInferer

class Trainer(object):
    def __init__(self):

        self.batch_size = cfg.batch_size
        self.strategy = cfg.strategy
        self.max_epochs = cfg.max_epochs

        self.summary_steps = cfg.summary_steps
        self.image_summary_step = cfg.image_summary_step

        self.save_step = cfg.save_step

        self.log_dir = cfg.log_dir
        self.tensorboard_writer = TensorboardWriter(self.log_dir)

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

        self.aster_ocr = AsterInferer()

        self.training_step = TrainingStep(
            self.generator,
            self.discriminator,
                self.aster_ocr,
            self.g_optimizer,
            self.d_optimizer,
            self.g_opt["reg_interval"],
            self.d_opt["reg_interval"],
            self.batch_size,
            self.pl_mean,
        )

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

            # start actual training
            print("Start Training")

            #setup loss trackers

            loss_trackers = [LossTracker(print_step, log_losses) for print_step, log_losses in zip(self.summary_steps["print_steps"], self.summary_steps["log_losses"])]

            # start training
            for real_images, input_texts, labels in dataset:
                step = self.g_optimizer.iterations.numpy()

                # g train step
                do_r1_reg = (step + 1) % self.d_opt["reg_interval"] == 0
                do_pl_reg = (step + 1) % self.g_opt["reg_interval"] == 0

                gen_losses, disc_losses, ocr_loss = self.training_step.dist_train_step(
                    real_images, input_texts, labels, do_r1_reg, do_pl_reg
                )
                reg_g_loss, g_loss, pl_penalty = gen_losses
                reg_d_loss, d_loss, r1_penalty = disc_losses

                # update g_clone
                self.g_clone.set_as_moving_average_of(self.generator)

                # get current step
                step = self.g_optimizer.iterations.numpy()

                losses_dict = {
                        "reg_g_loss": reg_g_loss,
                        "g_loss": g_loss,
                        "pl_penalty": pl_penalty,
                        "ocr_loss": ocr_loss,
                        "reg_d_loss": reg_d_loss,
                        "d_loss": d_loss,
                        "r1_penalty": r1_penalty
                        }

                for loss_tracker in loss_trackers:
                    loss_tracker.increment_losses(losses_dict)

                # save every self.save_step
                if step % self.save_step == 0:
                    self.manager.save(checkpoint_number=step)

                # save every self.image_summary_step
                if step % self.image_summary_step == 0:
                    self.tensorboard_writer.log_images(input_texts, self.g_clone, self.aster_ocr, step)

                # print every self.print_steps
                for loss_tracker in loss_trackers:
                    if step % loss_tracker.print_step == 0:
                        loss_tracker.print_losses(step)
                        if loss_tracker.log_losses:
                            self.tensorboard_writer.log_scalars(loss_tracker.losses, step)
                        loss_tracker.reinitialize_tracker()

            # save last checkpoint
            step = self.g_optimizer.iterations.numpy()
            self.manager.save(checkpoint_number=step)
            return

def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()
