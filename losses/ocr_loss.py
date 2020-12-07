import tensorflow as tf
from tensorflow import keras

class SoftmaxCrossEntropy(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='ocr_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)

