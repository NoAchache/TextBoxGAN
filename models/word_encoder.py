import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    ReLU,
)

from config import cfg


class WordEncoder(tf.keras.Model):
    def __init__(self, dense_dim=256, dropout_rate=0.3, name="word_encoder"):
        super(WordEncoder, self).__init__(name=name)

        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.max_chars = cfg.max_chars

        self.embedding_in_dim = len(cfg.char_tokenizer.main.word_index)
        self.embedding_out_dim = cfg.embedding_out_dim

        self.dropout = Dropout(self.dropout_rate)
        self.bilstm = Bidirectional(LSTM(128, return_sequences=True))
        self.fc = Dense(self.dense_dim)
        self.relu = ReLU()

        self.out_char_height = cfg.generator_resolutions[0][0]
        self.out_width = cfg.generator_resolutions[0][1]
        self.out_channels = cfg.generator_feat_maps[0]

    def build(self, input_shape):
        weight_shape = [self.embedding_in_dim - 1, self.embedding_out_dim]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w_embedding = tf.Variable(w_init, name="w_embedding", trainable=True)

        # embedding for the padding of labels
        w0_embedding = tf.zeros([1, self.embedding_out_dim])
        self.w0_embedding = tf.Variable(
            w0_embedding, name="w0_embedding", trainable=False
        )

    def call(self, inputs, batch_size=None, training=None, mask=None):

        input_texts = inputs

        w_embedding = tf.concat([self.w0_embedding, self.w_embedding], axis=0)
        embeddings = tf.nn.embedding_lookup(
            w_embedding, input_texts
        )  # (bs, max_chars, embedding_dim)
        embeddings = self.dropout(embeddings)
        x = self.bilstm(embeddings)  # (bs, max_chars, 128*2)

        x = tf.reshape(x, [batch_size * self.max_chars, 128 * 2])

        x = self.relu(self.fc(x))  # (bs * max_chars, self.dense_dim)

        x = tf.reshape(
            x,
            [batch_size, self.out_channels, self.out_char_height, self.out_width],
        )

        return x
