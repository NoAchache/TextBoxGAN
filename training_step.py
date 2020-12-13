import tensorflow as tf

from config import cfg
from models.losses.gan_losses import generator_loss, discriminator_loss
from models.losses.ocr_loss import softmax_cross_entropy_loss


class TrainingStep:
    def __init__(
        self,
        generator,
        discriminator,
        aster_ocr,
        g_optimizer,
        d_optimizer,
        g_reg_interval,
        d_reg_interval,
        batch_size,
        pl_mean,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.aster_ocr = aster_ocr
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_reg_interval = g_reg_interval
        self.d_reg_interval = d_reg_interval
        self.batch_size = batch_size
        self.pl_mean = pl_mean

        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_decay = 0.01
        self.r1_gamma = 10.0
        self.ocr_loss_weight = 1
        self.z_dim = cfg.z_dim
        self.char_width = cfg.char_width
        self.pl_noise_scaler = tf.math.rsqrt(
            float(cfg.im_width) * float(cfg.char_height)
        )

    @tf.function
    def dist_train_step(self, real_images, input_texts, labels, do_r1_reg, do_pl_reg):
        gen_losses, disc_losses, ocr_loss = cfg.strategy.experimental_run_v2(
            fn=self._train_step,
            args=(real_images, input_texts, labels, do_r1_reg, do_pl_reg),
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

    def _train_step(self, real_images, input_texts, labels, do_r1_reg, do_pl_reg):
        with tf.GradientTape() as g_tape:
            with tf.GradientTape() as d_tape:
                z = tf.random.normal(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
                fake_images = self.generator([input_texts, z], training=True)

                mask = tf.tile(
                    tf.expand_dims(
                        tf.expand_dims(
                            tf.repeat(
                                tf.where(input_texts == 0, 0.0, 1.0),
                                repeats=[self.char_width],
                                axis=1,
                            ),
                            1,
                        ),
                        1,
                    ), [1, fake_images.shape[1], fake_images.shape[2], 1]
                )

                fake_images = fake_images * mask

                fake_scores, reg_g_loss, g_loss, pl_penalty = self._get_gen_losses(
                    fake_images, do_pl_reg, input_texts
                )
                reg_d_loss, d_loss, r1_penalty = self._get_disc_losses(
                    fake_scores, real_images, do_r1_reg
                )

            ocr_loss = self._get_ocr_loss(fake_images, labels)
        # TODO: scale losses + add ocr_loss
            ocr_loss = self.ocr_loss_weight * ocr_loss

        g_gradients = g_tape.gradient(
            [reg_g_loss, ocr_loss], self.generator.trainable_variables
        )
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        d_gradients = d_tape.gradient(
            reg_d_loss, self.discriminator.trainable_variables
        )
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        gen_losses = (reg_g_loss, g_loss, pl_penalty)
        disc_losses = (reg_d_loss, d_loss, r1_penalty)

        return gen_losses, disc_losses, ocr_loss

    def _get_disc_losses(self, fake_scores, real_images, do_r1_reg):

        if do_r1_reg:
            real_scores, r1_penalty = self._r1_reg(real_images)
        else:
            real_scores = self.discriminator(real_images, training=True)
            r1_penalty = tf.constant(0.0, dtype=tf.float32)

        d_loss = discriminator_loss(fake_scores, real_scores)
        reg_d_loss = d_loss + r1_penalty

        return reg_d_loss, d_loss, r1_penalty

    def _get_gen_losses(self, fake_images, do_pl_reg, input_texts):
        fake_scores = self.discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_scores)

        pl_penalty = (
            self._path_length_reg(input_texts) if do_pl_reg else tf.constant(0.0, dtype=tf.float32)
        )
        reg_g_loss = g_loss + pl_penalty

        return fake_scores, reg_g_loss, g_loss, pl_penalty

    def _path_length_reg(self, input_texts):
        pl_minibatch = tf.maximum(
            1, tf.math.floordiv(self.batch_size, self.pl_minibatch_shrink)
        )
        pl_z = tf.random.normal(shape=[pl_minibatch, self.z_dim], dtype=tf.float32)


        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        with tf.GradientTape() as pl_tape:
            pl_tape.watch(pl_z)
            pl_fake_images, pl_style = self.generator(
                    (input_texts[:pl_minibatch], pl_z), ret_style=True, training=True
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
        return tf.reduce_sum(pl_penalty) / self.batch_size  # scales penalty

    def _r1_reg(self, real_images):
        with tf.GradientTape() as r1_tape:
            r1_tape.watch(real_images)
            real_scores = self.discriminator(real_images, training=True)
            real_loss = tf.reduce_sum(real_scores)

        real_grads = r1_tape.gradient(real_loss, real_images)
        r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
        r1_penalty = tf.expand_dims(r1_penalty, axis=1)
        r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_reg_interval
        r1_penalty = tf.reduce_sum(r1_penalty) / self.batch_size  # scales penalty
        return real_scores, r1_penalty

    def _get_ocr_loss(self, fake_images, labels):
        ocr_input_image = self.aster_ocr.convert_inputs(fake_images, labels, blank_label=1)
        logits = self.aster_ocr(ocr_input_image)
        return softmax_cross_entropy_loss(logits, labels)



