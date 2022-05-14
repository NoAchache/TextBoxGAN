import tensorflow as tf

from config import cfg

""" Available losses to teach the model to write the expected words. """


def softmax_cross_entropy_loss(y_pred: tf.float32, y_true: tf.float32) -> tf.float32:
    """Compares the OCR output of the generated image with the ground truth text."""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_sum(loss) / cfg.batch_size


def mean_squared_loss(
    y_with_noise: tf.float32, y_without_noise: tf.float32
) -> tf.float32:
    """Compares the OCR outputs of the generated and the real image, i.e. it teaches the network to write words as they
    are on the real images."""
    loss = tf.keras.losses.mse(y_with_noise, y_without_noise)
    return tf.reduce_sum(loss) / cfg.batch_size
