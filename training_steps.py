import tensorflow as tf

from config import train_cfg as cfg

from losses.stylegan2_losses import (
    d_logistic,
    d_logistic_r1_reg,
    g_logistic_non_saturating,
    g_logistic_ns_pathreg,
)


class TrainingSteps:
    def __init__(
        self,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        g_reg_interval,
        d_reg_interval,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_reg_interval = g_reg_interval
        self.d_reg_interval = d_reg_interval

        self.batch_scaler = 1.0 / float(cfg.batch_size)
        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_denorm = tf.math.rsqrt(
            float(256.0) * float(256.0)
        )  # TODO: 256.0 = train_res
        self.pl_decay = 0.01
        self.r1_gamma = 10.0

    def dist_d_train_step(self, inputs):
        per_replica_losses = cfg.strategy.experimental_run_v2(
            fn=self._d_train_step, args=(inputs,)
        )

        mean_d_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        return mean_d_loss

    def dist_d_train_step_reg(self, inputs):
        per_replica_losses = cfg.strategy.experimental_run_v2(
            fn=self._d_train_step_reg, args=(inputs,)
        )
        mean_d_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None
        )
        mean_d_gan_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None
        )
        mean_r1_penalty = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None
        )
        return mean_d_loss, mean_d_gan_loss, mean_r1_penalty

    def dist_g_train_step(self, inputs):
        per_replica_losses = cfg.strategy.experimental_run_v2(
            fn=self._g_train_step, args=(inputs,)
        )
        mean_g_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        return mean_g_loss

    def dist_g_train_step_reg(self, inputs):
        per_replica_losses = cfg.strategy.experimental_run_v2(
            fn=self._g_train_step_reg, args=(inputs,)
        )
        mean_g_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None
        )
        mean_g_gan_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None
        )
        mean_pl_penalty = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None
        )
        return mean_g_loss, mean_g_gan_loss, mean_pl_penalty

    def _d_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_loss = d_logistic(
                real_images, self.generator, self.discriminator, cfg.z_dim
            )

            # scale loss
            d_loss = tf.reduce_sum(d_loss) * self.batch_scaler

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        return d_loss

    def _d_train_step_reg(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as d_tape:
            # compute losses
            d_gan_loss, r1_penalty = d_logistic_r1_reg(
                real_images, self.generator, self.discriminator, cfg.z_dim
            )
            r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_reg_interval

            # scale losses
            r1_penalty = tf.reduce_sum(r1_penalty) * self.batch_scaler
            d_gan_loss = tf.reduce_sum(d_gan_loss) * self.batch_scaler

            # combine
            d_loss = d_gan_loss + r1_penalty

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        return d_loss, d_gan_loss, r1_penalty

    def _g_train_step(self, dist_inputs):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_loss = g_logistic_non_saturating(
                real_images, self.generator, self.discriminator, cfg.z_dim
            )

            # scale loss
            g_loss = tf.reduce_sum(g_loss) * self.batch_scaler

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        return g_loss

    def _g_train_step_reg(self, dist_inputs, pl_mean):
        real_images = dist_inputs[0]

        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pl_penalty = g_logistic_ns_pathreg(
                real_images,
                self.generator,
                self.discriminator,
                cfg.z_dim,
                pl_mean,
                self.pl_minibatch_shrink,
                self.pl_denorm,
                self.pl_decay,
            )
            pl_penalty = pl_penalty * self.pl_weight * self.g_reg_interval

            # scale loss
            pl_penalty = tf.reduce_sum(pl_penalty) * self.batch_scaler
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.batch_scaler

            # combine
            g_loss = g_gan_loss + pl_penalty

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        return g_loss, g_gan_loss, pl_penalty
