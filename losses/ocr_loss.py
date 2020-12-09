import tensorflow as tf
from tensorflow import keras


class SoftmaxCrossEntropyLoss(keras.losses.Loss):
    def __init__(
        self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="ocr_loss"
    ):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_pred, y_true):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
