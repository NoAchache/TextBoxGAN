import tensorflow as tf
from models.stylegan2.utils import lerp

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
        ret_style=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        training=None,
        mask=None,
    ):
        input_texts, z_latent = inputs  # ((bs, max_chars), (bs , z_dim))
        style = self.latent_encoder(z_latent, training=training)  # (bs, self.n_style, style_dim)
        word_encoded = self.word_encoder([input_texts, style[: self.n_style_w_e]], training=training)

        image_out = self.synthesis([word_encoded, style[-self.n_style_s :]], training=training)

        if ret_style:
            return image_out, style
        else:
            return image_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]

    @tf.function
    def set_as_moving_average_of(self, src_net):
        beta, beta_nontrainable = 0.99, 0.0

        for cw, sw in zip(self.weights, src_net.weights):
            assert sw.shape == cw.shape
            # print('{} <=> {}'.format(cw.name, sw.name))

            if "w_avg" in cw.name:
                cw.assign(lerp(sw, cw, beta_nontrainable))
            else:
                cw.assign(lerp(sw, cw, beta))
