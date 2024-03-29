import tensorflow as tf

from models.custom_stylegan2.layers.bias_act import BiasAct
from models.custom_stylegan2.layers.dense import Dense


class Mapping(tf.keras.layers.Layer):
    def __init__(self, style_dim, n_mapping, name, **kwargs):
        super(Mapping, self).__init__(name=name, **kwargs)
        self.style_dim = style_dim
        self.n_mapping = n_mapping
        self.gain = 1.0
        self.lrmul = 0.01

        self.normalize = tf.keras.layers.Lambda(
            lambda x: x
            * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)
        )

        self.dense_layers = list()
        self.bias_act_layers = list()
        for ii in range(self.n_mapping):
            self.dense_layers.append(
                Dense(
                    self.style_dim,
                    gain=self.gain,
                    lrmul=self.lrmul,
                    name="dense_{:d}".format(ii),
                )
            )
            self.bias_act_layers.append(
                BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_{:d}".format(ii))
            )

    def call(self, inputs):

        # normalize inputs
        x = self.normalize(inputs)

        # apply mapping blocks
        for dense, apply_bias_act in zip(self.dense_layers, self.bias_act_layers):
            x = dense(x)
            x = apply_bias_act(x)

        return x

    def get_config(self):
        config = super(Mapping, self).get_config()
        config.update(
            {
                "style_dim": self.style_dim,
                "n_mapping": self.n_mapping,
                "gain": self.gain,
                "lrmul": self.lrmul,
            }
        )
        return config
