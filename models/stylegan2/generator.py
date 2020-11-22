import tensorflow as tf

from models.stylegan2.layers.synthesis_block import Synthesis
from models.word_encoder import WordEncoder
from models.stylegan2.latent_encoder import LatentEncoder


class Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.word_encoder = WordEncoder()
        self.synthesis = Synthesis()
        self.n_style_w_e = 2 * len(self.word_encoder.char_expander.synth_blocks)
        self.n_style_s = 2 * len(self.synthesis.synth_blocks) + len(
            self.synthesis.torgbs
        )
        self.n_style = self.n_style_w_e + self.n_style_s
        self.latent_encoder = LatentEncoder(n_broadcast=self.n_style)

    def call(
        self,
        inputs,
        ret_w_broadcasted=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        training=None,
        mask=None,
    ):
        word_code, z_latent = inputs  # ((bs, max_chars), (bs , z_dim))
        style = self.latent_encoder(z_latent)  # (bs, self.n_style, w_dim)
        word_encoded = self.word_encoder([word_code, style[: self.n_style_w_e]])

        image_out = self.synthesis([word_encoded, style[-self.n_style_s :]])

        if ret_w_broadcasted:
            return (
                image_out,
                word_encoded,
            )  # w_broadcasted #TODO: dans quel cas c'est appelé?
        else:
            return image_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]
