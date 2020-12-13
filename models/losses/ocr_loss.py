import tensorflow as tf
from config import cfg

def softmax_cross_entropy_loss(y_pred, y_true):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_sum(loss) / cfg.global_batch_size
