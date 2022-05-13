from functools import partial

import tensorflow as tf

from models.stylegan2.layers.commons import compute_runtime_coef
from models.stylegan2.layers.dense import Dense
from models.stylegan2.layers.bias_act import BiasAct
from models.stylegan2.layers.cuda.upfirdn_2d_v2 import (
    upsample_conv_2d,
    compute_paddings,
)
from models.stylegan2.utils import apply_conv_in_good_format

class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_fmaps,
        out_fmaps,
        kernel_shape,
        up,
        demodulate,
        resample_kernel,
        gain,
        lrmul,
        fused_modconv,
        in_h_res=None,
        in_w_res=None,
        **kwargs
    ):
        super(ModulatedConv2D, self).__init__(**kwargs)

        self.in_fmaps = in_fmaps
        self.out_fmaps = out_fmaps
        self.kernel_shape = kernel_shape
        self.demodulate = demodulate
        self.up = up
        self.fused_modconv = fused_modconv
        self.in_h_res = in_h_res
        self.in_w_res = in_w_res
        self.gain = gain
        self.lrmul = lrmul

        self.k, self.pad0, self.pad1 = compute_paddings(
            resample_kernel, up, False, is_conv=True
        )

        # self.factor = 2
        self.mod_dense = Dense(self.in_fmaps, gain=1.0, lrmul=1.0, name="mod_dense")
        self.mod_bias = BiasAct(lrmul=1.0, act="linear", name="mod_bias")

    def build(self, input_shape):
        # x_shape, w_shape = input_shape[0], input_shape[1]
        weight_shape = self.kernel_shape + [self.in_fmaps, self.out_fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(
            weight_shape, self.gain, self.lrmul
        )

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name="w", trainable=True)

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        # height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare weights: [BkkIO] Introduce minibatch dimension
        w = self.runtime_coef * self.w
        ww = w[tf.newaxis]

        # Modulate
        s = self.mod_dense(y)  # [BI]
        s = self.mod_bias(s) + 1.0  # [BI]
        ww *= s[:, tf.newaxis, tf.newaxis, :, tf.newaxis]  # [BkkIO]

        if self.demodulate:
            d = tf.math.rsqrt(
                tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + 1e-8
            )  # [BO]
            ww *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # [BkkIO]

        if self.fused_modconv:
            # Fused => reshape minibatch to convolution groups
            x_shape = tf.shape(x)
            ww_shape = tf.shape(ww)
            x = tf.reshape(x, [1, -1, x_shape[2], x_shape[3]])  # [1, B*C, H, W]
            w = tf.reshape(
                tf.transpose(ww, [1, 2, 3, 0, 4]),
                [ww_shape[1], ww_shape[2], ww_shape[3], -1],
            )  # [k, k, I, B*O]
        else:
            # [BIhw] Not fused => scale input activations
            x *= s[:, :, tf.newaxis, tf.newaxis]

        # Convolution with optional upsampling.
        if self.up:
            x = upsample_conv_2d(
                x,
                self.in_w_res,
                self.in_h_res,
                w,
                self.pad0,
                self.pad1,
                self.k,
            )
        else:
            partial_conv_func = partial(tf.nn.conv2d,
                         filters=w, padding="SAME"
                    )
            x = apply_conv_in_good_format(x, partial_conv_func, h_w_stride=[1,1])


        # Reshape/scale output
        if self.fused_modconv:
            # Fused => reshape convolution groups back to minibatch
            x_shape = tf.shape(x)
            x = tf.reshape(x, [-1, self.out_fmaps, x_shape[2], x_shape[3]])
        elif self.demodulate:
            # [BOhw] Not fused => scale output activations
            x *= d[:, :, tf.newaxis, tf.newaxis]
        return x

    """
          output[b, h, w, BO] =
          sum_{k, k, C_in} input[b,  h + k, w + k, BC] *
                          filter[k, k, C_in, BO]

    """

    def get_config(self):
        config = super(ModulatedConv2D, self).get_config()
        config.update(
            {
                "in_w_res": self.in_w_res,
                "in_h_res": self.in_h_res,
                "in_fmaps": self.in_fmaps,
                "out_fmaps": self.out_fmaps,
                "kernel_shape": self.kernel_shape,
                "demodulate": self.demodulate,
                "fused_modconv": self.fused_modconv,
                "gain": self.gain,
                "lrmul": self.lrmul,
                "up": self.up,
                "k": self.k,
                "pad0": self.pad0,
                "pad1": self.pad1,
                "runtime_coef": self.runtime_coef,
            }
        )
        return config
