ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth

    allow_memory_growth()

import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from config.config import print_config
from dataset_utils.training_data_loader import TrainingDataLoader
from dataset_utils.validation_data_loader import ValidationDataLoader
from models.model_loader import ModelLoader
from training_step import TrainingStep
from utils import LossTracker, TensorboardWriter
from validation_step import ValidationStep


class Trainer(object):
    """Train the model. The different configs can be tuned in config/config."""

    def __init__(self):

        self.batch_size = cfg.batch_size
        self.strategy = cfg.strategy
        self.max_steps = cfg.max_steps
        self.summary_steps_frequency = cfg.summary_steps_frequency
        self.image_summary_step_frequency = cfg.image_summary_step_frequency
        self.save_step_frequency = cfg.save_step_frequency
        self.log_dir = cfg.log_dir

        self.validation_step_frequency = cfg.validation_step_frequency
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
        self.training_data_loader = TrainingDataLoader()
        self.validation_data_loader = ValidationDataLoader("validation_corpus.txt")
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
        self.ocr_optimizer = tf.keras.optimizers.Adam(
            self.g_opt["learning_rate"],
            beta_1=self.g_opt["beta1"],
            beta_2=self.g_opt["beta2"],
            epsilon=self.g_opt["epsilon"],
        )
        self.ocr_loss_weight = cfg.ocr_loss_weight

        self.aster_ocr = AsterInferer()

        self.training_step = TrainingStep(
            self.generator,
            self.discriminator,
            self.aster_ocr,
            self.g_optimizer,
            self.ocr_optimizer,
            self.d_optimizer,
            self.g_opt["reg_interval"],
            self.d_opt["reg_interval"],
            self.pl_mean,
        )

        self.validation_step = ValidationStep(self.g_clone, self.aster_ocr)

        self.manager = self.model_loader.load_checkpoint(
            ckpt_kwargs={
                "d_optimizer": self.d_optimizer,
                "g_optimizer": self.g_optimizer,
                "ocr_optimizer": self.ocr_optimizer,
                "discriminator": self.discriminator,
                "generator": self.generator,
                "g_clone": self.g_clone,
                "pl_mean": self.pl_mean,
            },
            model_description="Full model",
            expect_partial=False,
            ckpt_dir=cfg.ckpt_dir,
            max_to_keep=cfg.num_ckpts_to_keep,
        )

    @staticmethod
    def update_optimizer_params(params: dict):
        """
        Updates the optimizer configurations.

        Parameters
        ----------
        params: Configs of the optimizer

        Returns
        -------
        Updated configuration of the optimizer

        """
        params_copy = params.copy()
        mb_ratio = params_copy["reg_interval"] / (params_copy["reg_interval"] + 1)
        params_copy["learning_rate"] = params_copy["learning_rate"] * mb_ratio
        params_copy["beta1"] = params_copy["beta1"] ** mb_ratio
        params_copy["beta2"] = params_copy["beta2"] ** mb_ratio
        return params_copy

    def train(self):
        """
        Main training loop.

        """
        train_dataset = self.training_data_loader.load_dataset(
            batch_size=self.batch_size
        )

        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        validation_dataset = self.validation_data_loader.load_dataset(
            batch_size=self.batch_size
        )
        validation_dataset = self.strategy.experimental_distribute_dataset(
            validation_dataset
        )

        # start actual training
        print("Start Training")

        # setup loss trackers

        train_losses = [
            "reg_g_loss",
            "g_loss",
            "pl_penalty",
            "ocr_loss",
            "reg_d_loss",
            "d_loss",
            "r1_penalty",
        ]

        loss_trackers = [
            LossTracker(train_losses, print_step, log_losses)
            for print_step, log_losses in zip(
                self.summary_steps_frequency["print_steps"],
                self.summary_steps_frequency["log_losses"],
            )
        ]

        validation_tracker = LossTracker(["validation_ocr_loss"])
        # start training
        for real_images, ocr_image, input_words, ocr_labels in train_dataset:
            step = self.g_optimizer.iterations.numpy()

            # g train step
            do_r1_reg = True if (step + 1) % self.d_opt["reg_interval"] == 0 else False
            do_pl_reg = True if (step + 1) % self.g_opt["reg_interval"] == 0 else False

            if (
                step > 5000
            ):  # Set the ocr_loss_weight (close) to 0 at the beginning of the training since it is too early
                # to have a text to read from
                ocr_loss_weight = self.ocr_loss_weight

            else:
                ocr_loss_weight = 1e-8

            (gen_losses, disc_losses, ocr_loss,) = self.training_step.dist_train_step(
                real_images,
                ocr_image,
                input_words,
                ocr_labels,
                do_r1_reg,
                do_pl_reg,
                ocr_loss_weight,
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
                "r1_penalty": r1_penalty,
            }

            for loss_tracker in loss_trackers:
                loss_tracker.increment_losses(losses_dict)

            # save every self.save_step
            if step % self.save_step_frequency == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step_frequency == 0:
                self.tensorboard_writer.log_images(
                    input_words, self.g_clone, self.aster_ocr, step
                )

            if step % self.validation_step_frequency == 0:
                for input_words, ocr_labels in validation_dataset:
                    ocr_loss = self.validation_step.dist_validation_step(
                        input_words, ocr_labels
                    )
                    validation_tracker.increment_losses(
                        {"validation_ocr_loss": ocr_loss}
                    )

                self.tensorboard_writer.log_scalars(validation_tracker.losses, step)
                validation_tracker.print_losses(step)
                validation_tracker.reinitialize_tracker()

            # print every self.print_steps
            for loss_tracker in loss_trackers:
                if step % loss_tracker.print_step == 0:
                    loss_tracker.print_losses(step)
                    if loss_tracker.log_losses:
                        self.tensorboard_writer.log_scalars(loss_tracker.losses, step)
                    loss_tracker.reinitialize_tracker()
            if step == self.max_steps:
                break

        # save last checkpoint
        step = self.g_optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return


if __name__ == "__main__":
    print_config(cfg)
    with cfg.strategy.scope():
        trainer = Trainer()
        trainer.train()
