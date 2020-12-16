import tensorflow as tf
from config import cfg


def generator_loss(y_pred):
    loss = tf.math.softplus(-y_pred)
    return tf.reduce_sum(loss) / cfg.batch_size


def discriminator_loss(y_pred, y_true):
    loss = tf.math.softplus(y_pred)
    loss += tf.math.softplus(-y_true)
    return tf.reduce_sum(loss) / cfg.batch_size
