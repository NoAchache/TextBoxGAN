import tensorflow as tf
from tensorflow import keras

# TODO: check size is good
class GeneratorLoss(keras.losses.Loss):
    def __init__(
        self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="gen_loss"
    ):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_pred, y_true=None):
        return tf.math.softplus(y_pred)


class DiscriminatorLoss(keras.losses.Loss):
    def __init__(
        self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="disc_loss"
    ):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_pred, y_true):
        d_loss = tf.math.softplus(y_pred)
        d_loss += tf.math.softplus(-y_true)
        return d_loss
