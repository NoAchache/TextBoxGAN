import tensorflow as tf
import tensorflow_addons as tfa

from config import cfg
from utils import ASTER_CHAR_VECTOR


class AsterInferer:
    def __init__(self):
        tfa.register_all()
        self.model = tf.saved_model.load(cfg.aster_weights, tags='serve').signatures['serving_default']

    def run(self, image):

        prediction=self.model(image)
        return tf.map_fn(lambda t: ASTER_CHAR_VECTOR[t - 2] if t != 1 else "~",elems=prediction['recognition_text'],dtype=tf.string)
