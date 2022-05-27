from typing import List, Tuple

import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from models.custom_stylegan2.discriminator import Discriminator
from models.custom_stylegan2.generator import Generator
from models.losses.gan_losses import discriminator_loss, generator_loss
from models.losses.ocr_losses import mean_squared_loss, softmax_cross_entropy_loss
from utils.utils import mask_text_box


class TrainingStep:
    """Infer the model, computes the associated losses and backpropagates them."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        aster_ocr: AsterInferer,
        g_optimizer: tf.keras.optimizers.Adam,
        ocr_optimizer: tf.keras.optimizers.Adam,
        d_optimizer: tf.keras.optimizers.Adam,
        g_reg_interval: int,
        d_reg_interval: int,
        pl_mean: tf.float32,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.aster_ocr = aster_ocr
        self.g_optimizer = g_optimizer
        self.ocr_optimizer = ocr_optimizer
        self.d_optimizer = d_optimizer
        self.g_reg_interval = g_reg_interval
        self.d_reg_interval = d_reg_interval
        self.batch_size = cfg.batch_size
        self.batch_size_per_gpu = cfg.batch_size_per_gpu
        self.pl_mean = pl_mean

        pl_minibatch_shrink = 2
        self.pl_minibatch_shrink = (
            pl_minibatch_shrink
            if tf.math.floordiv(self.batch_size_per_gpu, pl_minibatch_shrink) >= 1
            else self.batch_size_per_gpu
        )
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_decay = 0.01
        self.r1_gamma = 10.0
        self.ocr_loss_type = cfg.ocr_loss_type
        self.z_dim = cfg.z_dim
        self.char_width = cfg.char_width
        self.pl_noise_scaler = tf.math.rsqrt(
            float(cfg.image_width) * float(cfg.char_height)
        )

    @tf.function
    def dist_train_step(
        self,
        real_images: tf.float32,
        ocr_images: tf.float32,
        input_words: tf.int32,
        ocr_labels: tf.int32,
        do_r1_reg: bool,
        do_pl_reg: bool,
        ocr_loss_weight: float,
    ) -> Tuple[
        Tuple["tf.float32", "tf.float32", "tf.float32"],
        Tuple["tf.float32", "tf.float32", "tf.float32"],
        "tf.float32",
    ]:
        """
        Entry point of the class. Distributes the training step on the available GPUs.

        Parameters
        ----------
        real_images: Real text boxes (i.e. from the dataset) preprocessed for our model.
        ocr_images: Real text boxes (i.e. from the dataset) preprocessed for the OCR model.
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        ocr_labels: Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.
        do_r1_reg: Whether to compute the R1 regression.
        do_pl_reg: Whether to compute the Path Length regression.
        ocr_loss_weight: Weight applied to the OCR loss.

        Returns
        -------
        Mean of the losses obtained for the text boxes generated from the input_words.

        """

        (gen_losses, disc_losses, ocr_loss,) = cfg.strategy.run(
            fn=self._train_step,
            args=(
                real_images,
                ocr_images,
                input_words,
                ocr_labels,
                do_r1_reg,
                do_pl_reg,
                ocr_loss_weight,
            ),
        )

        # Reduce generator losses
        reg_g_loss, g_loss, pl_penalty = gen_losses
        mean_reg_g_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, reg_g_loss, axis=None
        )
        mean_g_loss = cfg.strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)
        if do_pl_reg:
            mean_pl_penalty = cfg.strategy.reduce(
                tf.distribute.ReduceOp.SUM, pl_penalty, axis=None
            )
        else:
            mean_pl_penalty = tf.constant(0.0, dtype=tf.float32)

        mean_gen_losses = (mean_reg_g_loss, mean_g_loss, mean_pl_penalty)

        # Reduce discriminator losses
        reg_d_loss, d_loss, r1_penalty = disc_losses

        mean_reg_d_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, reg_d_loss, axis=None
        )
        mean_d_loss = cfg.strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
        mean_r1_penalty = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, r1_penalty, axis=None
        )

        mean_disc_losses = (mean_reg_d_loss, mean_d_loss, mean_r1_penalty)

        mean_ocr_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, ocr_loss, axis=None
        )

        return mean_gen_losses, mean_disc_losses, mean_ocr_loss

    def _train_step(
        self,
        real_images: tf.float32,
        ocr_images: tf.float32,
        input_words: tf.int32,
        ocr_labels: tf.int32,
        do_r1_reg: bool,
        do_pl_reg: bool,
        ocr_loss_weight: float,
    ) -> Tuple[
        Tuple["tf.float32", "tf.float32", "tf.float32"],
        Tuple["tf.float32", "tf.float32", "tf.float32"],
        "tf.float32",
    ]:
        """
        Generates text boxes from the input_words and compute their GAN and OCR losses.

        Parameters
        ----------
        real_images: Real text boxes (i.e. from the dataset) preprocessed for our model.
        ocr_images: Real text boxes (i.e. from the dataset) preprocessed for the OCR model.
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        ocr_labels: Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.
        do_r1_reg: Whether to compute the R1 regression.
        do_pl_reg: Whether to compute the Path Length regression.
        ocr_loss_weight: Weight applied to the OCR loss.

        Returns
        -------
        gen_losses: Losses associated to the generator.
        disc_losses: Losses associated to the discriminator.
        ocr_loss / ocr_loss_weight: weighted OCR loss

        """

        with tf.GradientTape() as ocr_tape:
            with tf.GradientTape() as g_tape:
                with tf.GradientTape() as d_tape:
                    z = tf.random.normal(
                        shape=[self.batch_size_per_gpu, self.z_dim],
                        dtype=tf.dtypes.float32,
                    )
                    fake_images = self.generator([input_words, z])

                    fake_images = mask_text_box(
                        fake_images, input_words, self.char_width
                    )

                    (
                        fake_scores,
                        reg_g_loss,
                        g_loss,
                        pl_penalty,
                    ) = self._get_generator_losses(fake_images, do_pl_reg, input_words)
                    reg_d_loss, d_loss, r1_penalty = self._get_discriminator_losses(
                        fake_scores, real_images, do_r1_reg
                    )

            ocr_loss = self._get_ocr_loss(fake_images, ocr_labels, ocr_images)
            ocr_loss = ocr_loss_weight * ocr_loss

        self._backpropagates_gradient(
            tape=g_tape,
            models=[self.generator.synthesis, self.generator.latent_encoder],
            loss=reg_g_loss,
            optimizer=self.g_optimizer,
        )

        self._backpropagates_gradient(
            tape=ocr_tape,
            models=[self.generator.synthesis, self.generator.word_encoder],
            loss=ocr_loss,
            optimizer=self.ocr_optimizer,
        )

        self._backpropagates_gradient(
            tape=d_tape,
            models=[self.discriminator],
            loss=reg_d_loss,
            optimizer=self.d_optimizer,
        )

        gen_losses = (reg_g_loss, g_loss, pl_penalty)
        disc_losses = (reg_d_loss, d_loss, r1_penalty)

        return (
            gen_losses,
            disc_losses,
            ocr_loss / ocr_loss_weight,
        )

    def _backpropagates_gradient(
        self,
        tape: tf.GradientTape,
        models: List[tf.keras.Model],
        loss: tf.float32,
        optimizer: tf.keras.optimizers.Adam,
    ) -> None:
        """Backpropagates the gradient of the loss into the given networks"""

        trainable_variables = sum([model.trainable_variables for model in models], [])
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    def _get_discriminator_losses(
        self, fake_scores: tf.float32, real_images: tf.float32, do_r1_reg: bool
    ) -> Tuple["tf.float32", "tf.float32", "tf.float32"]:
        """
        Computes the losses associated to the discriminator, i.e. the discriminator loss and the R1 regression

        Parameters
        ----------
        fake_scores: Output of the discriminator when inferring the fake_images.
        real_images: Real text boxes (i.e. from the dataset) preprocessed for our model.
        do_r1_reg: Whether to compute the R1 regression.

        Returns
        -------
        reg_d_loss: Regularized discriminator loss.
        d_loss: Discriminator loss.
        r1_penalty: Penalty of the Path Length regression.

        """

        if do_r1_reg:
            real_scores, r1_penalty = self._r1_reg(real_images)
        else:
            real_scores = self.discriminator(real_images)
            r1_penalty = tf.constant(0.0, dtype=tf.float32)

        d_loss = discriminator_loss(fake_scores, real_scores)
        reg_d_loss = d_loss + r1_penalty

        return reg_d_loss, d_loss, r1_penalty

    def _get_generator_losses(
        self, fake_images: tf.float32, do_pl_reg: bool, input_words: tf.int32
    ) -> Tuple["tf.float32", "tf.float32", "tf.float32", "tf.float32"]:
        """
        Computes the losses associated to the generator, i.e. the generator loss and the Path Length regression

        Parameters
        ----------
        fake_images: Text boxes generated with our model.
        do_pl_reg: Whether to compute the Path Length regression.
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.

        Returns
        -------
        fake_scores: Output of the discriminator when inferring the fake_images.
        reg_g_loss: Regularized generator loss.
        g_loss: Generator loss.
        pl_penalty: Penalty of the Path Length regression.

        """
        fake_scores = self.discriminator(fake_images)
        g_loss = generator_loss(fake_scores)

        pl_penalty = (
            self._path_length_reg(input_words)
            if do_pl_reg
            else tf.constant(0.0, dtype=tf.float32)
        )
        reg_g_loss = g_loss + pl_penalty

        return fake_scores, reg_g_loss, g_loss, pl_penalty

    def _path_length_reg(self, input_words) -> tf.float32:
        """
        Computes the Path Length regression.

        Parameters
        ----------
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.

        Returns
        -------
        Penalty of the Path Length regression.


        """
        pl_minibatch = tf.maximum(
            1, tf.math.floordiv(self.batch_size_per_gpu, self.pl_minibatch_shrink)
        )
        pl_z = tf.random.normal(
            shape=[pl_minibatch, self.z_dim],
            dtype=tf.dtypes.float32,
        )

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        with tf.GradientTape() as pl_tape:
            pl_tape.watch(pl_z)
            pl_fake_images, pl_style = self.generator(
                (input_words[:pl_minibatch], pl_z),
                batch_size=pl_minibatch,
                ret_style=True,
            )
            pl_noise = tf.random.normal(tf.shape(pl_fake_images)) * self.pl_noise_scaler
            pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

        pl_grads = pl_tape.gradient(pl_noise_applied, pl_style)
        pl_lengths = tf.math.sqrt(
            tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1)
        )
        # Track exponential moving average of |J*y|.
        pl_mean_val = self.pl_mean + self.pl_decay * (
            tf.reduce_mean(pl_lengths) - self.pl_mean
        )
        self.pl_mean.assign(pl_mean_val)

        # Calculate (|J*y|-a)^2.
        pl_penalty = tf.square(pl_lengths - self.pl_mean)

        pl_penalty = pl_penalty * self.pl_minibatch_shrink * self.g_reg_interval
        return tf.reduce_sum(pl_penalty) / self.batch_size  # scales penalty

    def _r1_reg(self, real_images: tf.float32) -> Tuple["tf.float32", "tf.float32"]:
        """
        Infer the discriminator and computes the R1 regression.

        Parameters
        ----------
        real_images: Real text boxes (i.e. from the dataset) preprocessed for our model.

        Returns
        -------
        real_scores: Output of the discriminator when inferring the real_images.
        r1_penalty: Penalty of the R1 regression.

        """
        with tf.GradientTape() as r1_tape:
            r1_tape.watch(real_images)
            real_scores = self.discriminator(real_images)
            real_loss = tf.reduce_sum(real_scores)

        real_grads = r1_tape.gradient(real_loss, real_images)
        r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
        r1_penalty = tf.expand_dims(r1_penalty, axis=1)
        r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_reg_interval
        r1_penalty = tf.reduce_sum(r1_penalty) / self.batch_size  # scales penalty
        return real_scores, r1_penalty

    def _get_ocr_loss(
        self, fake_images: tf.float32, ocr_labels: tf.int32, ocr_images: tf.float32
    ) -> tf.float32:
        """
        Computes the OCR loss.

        Parameters
        ----------
        fake_images: Text boxes generated with our model.
        ocr_labels: Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.
        ocr_images: Real text boxes (i.e. from the dataset) preprocessed for the OCR model.

        Returns
        -------
        OCR loss obtained for the fake_images.

        """

        fake_images_ocr_format = self.aster_ocr.convert_inputs(
            fake_images, ocr_labels, blank_label=1
        )
        logits = self.aster_ocr(fake_images_ocr_format)

        if self.ocr_loss_type == "mse":
            real_logits = self.aster_ocr(ocr_images)
            return mean_squared_loss(real_logits, logits)
        elif self.ocr_loss_type == "softmax_crossentropy":
            return softmax_cross_entropy_loss(logits, ocr_labels)
