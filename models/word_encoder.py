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
)


from config import cfg
from models.stylegan2.layers.synthesis_block import SynthesisBlock


class WordEncoder(tf.keras.Model):
    def __init__(self, name="word_encoder"):
        super(WordEncoder, self).__init__(name=name)

        self.encoding_dense_dim = 256
        self.char_encoder = CharEncoder(dense_dim=self.encoding_dense_dim)

        assert cfg.char_width % 4 == 0
        self.encoded_char_width = cfg.char_width / 4

        self.char_expander = CharExpander(
            encoded_char_width=self.encoded_char_width,
            init_channels=self.encoding_dense_dim / self.encoded_char_width,
        )

        self.fc = Sequential([Dense(512), BatchNormalization(), ReLU()])

    def call(self, inputs, training=None, mask=None):

        input_texts, style = inputs
        chars_encoded = self.char_encoder(
            input_texts
        )  # (bs * max_chars, self.encoding_dense_dim)

        word_encoded = self.char_expander([chars_encoded, style])
        # (bs, cfg.expand_char_feat_maps[-1], 1,cfg.image_width)

        return word_encoded


class CharEncoder(tf.keras.Model):
    def __init__(self, dense_dim, dropout_rate=0.3, name="char_encoder"):
        super(CharEncoder, self).__init__(name=name)

        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate

        self.embedding_in_dim = len(cfg.char_tokenizer.main.word_index)
        self.embedding_out_dim = cfg.embedding_out_dim

        self.dropout = Dropout(self.dropout_rate)
        self.bilstm = Bidirectional(LSTM(128, return_sequences=True))
        self.fc = Dense(self.dense_dim)
        self.relu = ReLU()

    def build(self, input_shape):
        weight_shape = [self.embedding_in_dim - 1, self.embedding_out_dim]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w_embedding = tf.Variable(w_init, name="w_embedding", trainable=True)

        # embedding for the padding of labels
        w0_embedding = tf.zeros([1, self.embedding_out_dim])
        self.w0_embedding = tf.Variable(
            w0_embedding, name="w0_embedding", trainable=False
        )

    def call(self, inputs, training=None, mask=None):
        w_embedding = tf.concat([self.w0_embedding, self.w_embedding], axis=0)
        embeddings = tf.nn.embedding_lookup(
            w_embedding, inputs
        )  # (bs, max_chars, embedding_dim)
        embeddings = self.dropout(embeddings)
        x = self.bilstm(embeddings)  # (bs, max_chars, 128*2)
        x = tf.reshape(x, [cfg.batch_size * cfg.max_chars, 128 * 2])
        x = self.relu(self.fc(x))  # (bs * max_chars, self.dense_dim)

        return x


class CharExpander(tf.keras.Model):
    def __init__(self, encoded_char_width, init_channels, name="char_expander"):
        super(CharExpander, self).__init__(name=name)

        self.width = encoded_char_width
        self.init_channels = init_channels
        self.width_resolutions = cfg.expand_char_w_res
        self.feat_maps = cfg.expand_char_feat_maps

        self.synth_blocks = list()
        self.feat_maps = [self.channels] + self.feat_maps

        for out_w, in_maps, out_maps in zip(
            self.width_resolutions, self.feat_maps[:-1], self.feat_maps[1:]
        ):
            self.synth_blocks.append(
                SynthesisBlock(
                    in_ch=in_maps,
                    out_fmaps=out_maps,
                    out_h_res=1,
                    out_w_res=out_w,
                    expand_direction="width",
                    kernel_shape=[1, 3],
                    name="{:d}x{:d}/block".format(1, out_w),
                )
            )

    def call(self, inputs, training=None, mask=None):
        x, style = inputs

        x = tf.reshape(
            inputs, [cfg.batch_size * cfg.max_chars, self.init_channels, 1, self.width],
        )

        for idx, block in enumerate(self.synth_blocks):
            idx *= 2
            s0 = style[:, idx]
            s1 = style[:, idx + 1]
            x = block([x, s0, s1])

        # x: (bs * max_chars, self.feat_maps[-1], 1, cfg.char_width)

        x = tf.reshape(
            x, [cfg.batch_size, self.feat_maps[-1], 1, cfg.max_chars * cfg.char_width]
        )

        return x
