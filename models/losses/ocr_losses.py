import tensorflow as tf

from config import cfg


def softmax_cross_entropy_loss(y_pred, y_true):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_sum(loss) / cfg.batch_size


def mean_squared_loss(y_with_noise, y_without_noise):
    loss = tf.keras.losses.mse(y_with_noise, y_without_noise)
    return tf.reduce_sum(loss) / cfg.batch_size
