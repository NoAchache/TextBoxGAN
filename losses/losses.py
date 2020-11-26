import tensorflow as tf
from config import cfg


class Losses:
    def __init__(self):
        self.batch_size = cfg.batch_size

    def generator_loss(self, fake_scores):
        g_loss = tf.math.softplus(-fake_scores)
        return tf.reduce_sum(g_loss) / self.batch_size  # scales the loss

    def discriminator_loss(self, fake_scores, real_scores):
        d_loss = tf.math.softplus(fake_scores)
        d_loss += tf.math.softplus(-real_scores)
        return tf.reduce_sum(d_loss) / self.batch_size  # scales the loss
