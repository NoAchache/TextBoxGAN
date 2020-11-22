import tensorflow as tf

from models.stylegan2.layers.commons import compute_runtime_coef
from models.stylegan2.layers.cuda.upfirdn_2d_v2 import (
    conv_downsample_2d,
    compute_paddings,
)


class Conv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_res,
        in_fmaps,
        fmaps,
        kernel,
        down,
        resample_kernel,
        gain,
        lrmul,
        **kwargs
    ):
        super(Conv2D, self).__init__(**kwargs)
        self.in_res = in_res
        self.in_fmaps = in_fmaps
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.down = down

        self.k, self.pad0, self.pad1 = compute_paddings(
            resample_kernel, self.kernel, False, down, is_conv=True
        )

    def build(self, input_shape):
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(
            weight_shape, self.gain, self.lrmul
        )

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name="w", trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        w = self.runtime_coef * self.w

        # actual conv
        if self.down:
            x = conv_downsample_2d(
                x,
                self.in_res,
                w,
                self.kernel,
                self.kernel,
                self.pad0,
                self.pad1,
                self.k,
            )

            # # TODO: ensure this works
            # x = conv_downsample_2d(
            #     x, self.in_w_res, self.in_h_res, w, self.pad0, self.pad1, self.k,
            # )
        else:
            x = tf.nn.conv2d(
                x, w, data_format="NCHW", strides=[1, 1, 1, 1], padding="SAME"
            )
        return x

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update(
            {
                "in_res": self.in_res,
                "in_fmaps": self.in_fmaps,
                "fmaps": self.fmaps,
                "kernel": self.kernel,
                "gain": self.gain,
                "lrmul": self.lrmul,
                "down": self.down,
                "k": self.k,
                "pad0": self.pad0,
                "pad1": self.pad1,
                "runtime_coef": self.runtime_coef,
            }
        )
        return config
