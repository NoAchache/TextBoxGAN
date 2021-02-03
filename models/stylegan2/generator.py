import tensorflow as tf

from config import cfg
from models.stylegan2.latent_encoder import LatentEncoder
from models.stylegan2.layers.synthesis_block import Synthesis
from models.stylegan2.utils import lerp
from models.word_encoder import WordEncoder


class Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.word_encoder = WordEncoder()
        self.synthesis = Synthesis()
        self.n_style = 2 * len(self.synthesis.synth_blocks) + len(self.synthesis.torgbs)
        self.latent_encoder = LatentEncoder(n_broadcast=self.n_style)

    def call(
        self,
        inputs,
        batch_size=cfg.batch_size_per_gpu,
        ret_style=False,
        truncation_psi=1.0,
        training=None,
        mask=None,
    ):
        input_words, z_latent = inputs  # ((bs, max_char_number), (bs , z_dim))

        word_encoded = self.word_encoder(
            input_words,
            batch_size=batch_size,
            training=training,
        )

        style = self.latent_encoder(
            z_latent, training=training, truncation_psi=truncation_psi
        )  # (bs, self.n_style, style_dim)

        image_out = self.synthesis([word_encoded, style], training=training)

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
