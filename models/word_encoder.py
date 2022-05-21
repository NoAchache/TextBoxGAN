import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, ReLU

from config import cfg


class WordEncoder(tf.keras.Model):
    """Encodes the word through dense layers while keeping the order of the letters."""

    def __init__(self, dropout_rate=0.3, name="word_encoder"):
        super(WordEncoder, self).__init__(name=name)

        self.dense_dim = cfg.word_encoder_dense_dim
        self.dropout_rate = dropout_rate
        self.max_char_number = cfg.max_char_number

        self.embedding_in_dim = len(cfg.char_tokenizer.main.word_index)
        self.embedding_out_dim = cfg.embedding_out_dim

        self.dropout = Dropout(self.dropout_rate)
        self.fc = Dense(self.dense_dim)
        self.relu = ReLU()

        self.out_char_height = cfg.generator_resolutions[0][0]
        self.out_width = cfg.generator_resolutions[0][1]
        self.out_channels = cfg.generator_feat_maps[0]

    def build(self, input_shape):
        weight_shape = [self.embedding_in_dim - 1, self.embedding_out_dim]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w_embedding = tf.Variable(w_init, name="w_embedding", trainable=True)

        # embedding for the padding added to words whose lengths are less than cfg.max_char_number
        w0_embedding = tf.zeros([1, self.embedding_out_dim])
        self.w0_embedding = tf.Variable(
            w0_embedding, name="w0_embedding", trainable=False
        )

    def call(self, inputs: tf.int32, batch_size=None):

        input_words = inputs

        w_embedding = tf.concat([self.w0_embedding, self.w_embedding], axis=0)
        embeddings = tf.nn.embedding_lookup(
            w_embedding, input_words
        )  # (bs, max_char_number, embedding_dim)
        embeddings = self.dropout(embeddings)

        x = tf.reshape(
            embeddings, [batch_size * self.max_char_number, self.embedding_out_dim]
        )

        x = self.relu(self.fc(x))  # (bs * max_char_number, self.dense_dim)

        x = tf.transpose(
            tf.reshape(
                x,
                [batch_size, self.out_width, self.out_channels, self.out_char_height],
            ),
            (0, 2, 3, 1),
        )

        return x
