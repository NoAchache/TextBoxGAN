import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    BatchNormalization,
    ReLU,
    Convolution2DTranspose,
)

from models.stylegan2.latent_encoder import LatentEncoder
from utils import NUM_CLASSES
from config import train_cfg as cfg


# embedding(32) --> dropout(0.7) --> biLSTM(256) --> fc(512) --> relu --> cat(z(100))--> reshape: batch size *= 8
# --> fc(512) --> batchnorm1d --> relu


class WordEncoder(tf.keras.Model):
    def __init__(self, name="WordEncoder"):
        super(WordEncoder, self).__init__(name=name)

        self.char_encoder = CharEncoder()
        self.latent_encoder = LatentEncoder()
        self.char_expander = CharExpander()

        self.fc = Sequential([Dense(512), BatchNormalization(), ReLU()])

    def call(self, inputs, training=None, mask=None):

        word_code, z_latent = inputs  # ((bs, max_chars), (bs * max_chars, z_dim_char))
        chars_encoded = self.char_encoder(word_code)  # (bs * max_chars, 256)
        w_latent = self.latent_encoder(z_latent)  # (bs * max_chars, w_dim_char)
        chars_w = tf.concat(
            [chars_encoded, w_latent], axis=1
        )  # (bs * max_chars, 256 + w_dim_char)

        chars_w = self.fc(chars_w)  # (bs * max_chars, 512)

        word_encoded = self.char_expander(chars_w)
        # (bs, 1, max_chars * char_width, c) with c = 512 * 16 / char_width

        return word_encoded


class CharEncoder(tf.keras.Model):
    def __init__(self, dropout_rate=0.3, name="CharEncoder"):
        super(CharEncoder, self).__init__(name=name)

        self.char_embedding = Embedding(
            NUM_CLASSES, cfg.embedding_dim, input_length=cfg.max_chars
        )

        self.dropout = Dropout(dropout_rate)
        self.bilstm = Bidirectional(LSTM(128, return_sequences=True))
        self.fc = Dense(256)
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        embeddings = self.dropout(
            self.char_embedding(inputs)  # (bs, max_chars, embedding_dim)
        )
        x = self.bilstm(embeddings)  # (bs, max_chars, 128*2)
        x = tf.reshape(x, [cfg.batch_size * cfg.max_chars, 128 * 2])
        x = self.relu(self.fc(x))  # (bs * max_chars, 256)

        return x


class CharExpander(tf.keras.Model):
    def __init__(self, name="CharExpander"):
        super(CharExpander, self).__init__(name=name)

        assert cfg.char_width % 4 == 0
        self.width = cfg.char_width / 4

    def _upBlock(self, output_filters, kernel_size):

        return Sequential(
            [
                Convolution2DTranspose(
                    output_filters, kernel_size, data_format="channels_last"
                ),
                BatchNormalization(),
                GLU(),
            ]
        )

    def build(self, input_shape):
        assert input_shape[1] % self.width == 0
        self.channels = input_shape[1] / self.width
        self.up_block1 = self._upBlock(
            output_filters=self.channels * 4, kernel_size=(1, 2)
        )
        self.up_block2 = self._upBlock(
            output_filters=self.channels * 4, kernel_size=(1, 2)
        )

        # TODO: replace upBlock with a stylegan2 upBlock

    def call(self, inputs, training=None, mask=None):

        x = tf.reshape(
            inputs, [cfg.batch_size * cfg.max_chars, 1, self.width, self.channels],
        )
        x = self.up_block1(x)  # (bs * max_chars, 1, cfg.char_width / 2, channels)
        x = self.up_block2(x)  # (bs * max_chars, 1, cfg.char_width, channels)

        # (bs * max_chars, 1, cfg.char_width, channels)
        x = tf.reshape(
            x, [cfg.batch_size, 1, cfg.max_chars * cfg.char_width, self.channels * 4]
        )

        return x


class GLU(tf.keras.layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, "channels dont divide 2!"
        self.output_dim = input_shape[-1] // 2

    def call(self, x, training=None, mask=None):
        nc = self.output_dim
        return x[:, :, :, :nc] * tf.sigmoid(x[:, :, :, nc:])
