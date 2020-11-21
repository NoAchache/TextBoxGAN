import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    BatchNormalization,
    ReLU,
)

from models.stylegan2.latent_encoder import LatentEncoder
from utils import NUM_CLASSES
from config import train_cfg as cfg


# embedding(32) --> dropout(0.7) --> biLSTM(256) --> fc(512) --> relu --> cat(z(100))--> reshape: batch size *= 8
# --> fc(512) --> batchnorm1d --> relu --> reshape en 4*4*C? --> conv jusqu'a 32*1*C
#                                     --> reshape en 4*1*(4*C)? : check dim au stylegan 2 (combien de channels?)


class WordEncoder(tf.keras.Model):
    def __init__(self, dropout_rate=0.3, name="WordEncoder"):
        super(WordEncoder, self).__init__(name=name)

        self.char_encoder = CharEncoder()
        self.latent_encoder = LatentEncoder()
        self.char_expander = CharExpander()

    def call(self, inputs, training=None, mask=None):
        # bs: batch size

        word_code, z_latent = inputs
        chars_encoded = self.char_encoder(word_code)  # (bs, 64 * max_chars)
        w_latent = self.latent_encoder(z_latent)  # (bs, z_dim_char * max_chars)

        # TODO: reshape chars encoded in (bs*max_chars, 4, 1, 16)
        # TODO: reshape w_latent in (bs*max_chars, 4, 1, z_dim_char/4)

        # TODO: cat the 2 above variables in the channels direction (ps: z_dim_char

        return embeddings


class CharEncoder(tf.keras.Model):
    def __init__(self, dropout_rate=0.3, name="CharEncoder"):
        super(CharEncoder, self).__init__(name=name)

        self.char_embedding = Embedding(
            NUM_CLASSES, cfg.embedding_dim, input_length=cfg.max_chars
        )
        fc_dim = 64 * cfg.max_chars
        self.dropout = Dropout(dropout_rate)
        self.bilstm = Bidirectional(LSTM(256))
        self.fc = Dense(fc_dim)
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):

        embeddings = self.dropout(self.char_embedding(inputs))
        x = self.bilstm(embeddings)
        x = self.relu(self.fc(x))

        return x


class CharExpander(tf.keras.Model):
    def __init__(self, name="CharExpander"):
        super(CharExpander, self).__init__(name=name)

        self.fc2 = Dense(512)
        self.batch_norm_1d = BatchNormalization()
        self.glu = GLU()

    def call(self, inputs, training=None, mask=None):
        embeddings = self.dropout(self.char_embedding(encoded_label))
        x = self.bilstm(embeddings)
        x = self.fc1(x)

        return embeddings
