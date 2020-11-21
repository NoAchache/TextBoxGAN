import tensorflow as tf

from models.stylegan2.layers.synthesis_block import Synthesis
from models.word_encoder import WordEncoder


class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.resolutions = g_params["resolutions"]
        self.featuremaps = g_params["featuremaps"]

        self.word_encoder = WordEncoder()
        self.synthesis = Synthesis(
            self.resolutions, self.featuremaps, name="g_synthesis"
        )

    def call(
        self,
        inputs,
        ret_w_broadcasted=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        training=None,
        mask=None,
    ):
        word_encoded = self.word_encoder(inputs)

        image_out = self.synthesis(word_encoded)

        if ret_w_broadcasted:
            return (
                image_out,
                word_encoded,
            )  # w_broadcasted #TODO: dans quel cas c'est appelé?
        else:
            return image_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]
