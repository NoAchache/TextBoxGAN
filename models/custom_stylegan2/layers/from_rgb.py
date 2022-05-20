import tensorflow as tf

from models.custom_stylegan2.layers.bias_act import BiasAct
from models.custom_stylegan2.layers.conv import Conv2D


class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, h_res, w_res, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.h_res = h_res
        self.w_res = w_res

        self.conv = Conv2D(
            in_fmaps=3,
            out_fmaps=self.fmaps,
            kernel=1,
            down=False,
            resample_kernel=None,
            gain=1.0,
            lrmul=1.0,
            name="conv",
        )
        self.apply_bias_act = BiasAct(lrmul=1.0, act="lrelu", name="bias")

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias_act(y)
        return y

    def get_config(self):
        config = super(FromRGB, self).get_config()
        config.update(
            {
                "fmaps": self.fmaps,
                "h_res": self.h_res,
                "w_res": self.w_res,
            }
        )
        return config
