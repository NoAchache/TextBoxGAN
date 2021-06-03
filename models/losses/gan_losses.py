import tensorflow as tf

from config import cfg

""" Usual GAN losses. Ensures the generated text box looks similar to real text boxes. """


def generator_loss(y_pred: tf.float32) -> tf.float32:
    loss = tf.math.softplus(-y_pred)
    return tf.reduce_sum(loss) / cfg.batch_size


def discriminator_loss(y_pred: tf.float32, y_true: tf.float32) -> tf.float32:
    loss = tf.math.softplus(y_pred)
    loss += tf.math.softplus(-y_true)
    return tf.reduce_sum(loss) / cfg.batch_size
