import tensorflow as tf

from models.stylegan2.layers.modulated_conv2d import ModulatedConv2D
from models.stylegan2.layers.bias_act import BiasAct
from models.stylegan2.layers.noise import Noise
from models.stylegan2.layers.to_rgb import ToRGB
from models.stylegan2.layers.cuda.upfirdn_2d_v2 import (
    upsample_height_2d,
    compute_paddings,
)
from config import cfg


class SynthesisBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_ch,
        out_fmaps,
        out_h_res,
        out_w_res,
        expand_direction,
        kernel_shape,
        **kwargs
    ):
        super(SynthesisBlock, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.fmaps = out_fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        self.out_h_res = out_h_res
        self.out_w_res = out_w_res
        self.expand_direction = expand_direction
        self.kernel_shape = kernel_shape

        # conv0 up
        self.conv_0 = ModulatedConv2D(
            in_fmaps=self.in_ch,
            out_fmaps=self.fmaps,
            kernel_shape=self.kernel_shape,
            up=True,
            demodulate=True,
            resample_kernel=[1, 3, 3, 1],
            gain=self.gain,
            lrmul=self.lrmul,
            fused_modconv=True,
            in_h_res=self.out_h_res // 2
            if self.expand_direction == "height"
            else self.out_h_res,
            in_w_res=self.out_w_res // 2
            if self.expand_direction == "width"
            else self.out_w_res,
            h_expand_factor=2 if self.expand_direction == "height" else 1,
            w_expand_factor=2 if self.expand_direction == "width" else 1,
            name="conv_0",
        )
        self.apply_noise_0 = Noise(name="noise_0")
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_0")

        # conv block
        self.conv_1 = ModulatedConv2D(
            in_w_res=self.out_w_res,
            in_h_res=self.out_h_res,
            in_fmaps=self.fmaps,
            out_fmaps=self.fmaps,
            kernel_shape=self.kernel_shape,
            up=False,
            demodulate=True,
            resample_kernel=[1, 3, 3, 1],
            gain=self.gain,
            lrmul=self.lrmul,
            fused_modconv=True,
            name="conv_1",
        )
        self.apply_noise_1 = Noise(name="noise_1")
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_1")

    def call(self, inputs, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_act_0(x)

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_act_1(x)
        return x

    def get_config(self):
        config = super(SynthesisBlock, self).get_config()
        config.update(
            {
                "in_ch": self.in_ch,
                "res": self.res,
                "fmaps": self.fmaps,
                "gain": self.gain,
                "lrmul": self.lrmul,
            }
        )
        return config


class Synthesis(tf.keras.layers.Layer):
    def __init__(self, name="synthesis", **kwargs):
        super(Synthesis, self).__init__(name=name, **kwargs)
        self.h_resolutions = cfg.expand_word_h_res
        self.feat_maps = cfg.expand_word_feat_maps
        self.width = cfg.im_width

        self.k, self.pad0, self.pad1 = compute_paddings(
            [1, 3, 3, 1], up=True, down=False, is_conv=False
        )

        self.initial_torgb = ToRGB(
            in_ch=self.feat_maps[0],
            h_res=self.h_resolutions[0],
            name="{:d}x{:d}/ToRGB".format(self.h_resolutions[0], self.width),
        )

        # stack generator block with lerp block
        prev_f_m = self.feat_maps[0]
        self.synth_blocks = list()
        self.torgbs = list()

        for h_res, f_m in zip(self.h_resolutions[1:], self.feat_maps[1:]):
            self.synth_blocks.append(
                SynthesisBlock(
                    in_ch=prev_f_m,
                    out_fmaps=f_m,
                    out_h_res=h_res,
                    out_w_res=self.width,
                    expand_direction="height",
                    kernel_shape=[3, 3],
                    name="{:d}x{:d}/block".format(h_res, self.width),
                )
            )

            self.torgbs.append(
                ToRGB(
                    in_ch=f_m,
                    h_res=h_res,
                    name="{:d}x{:d}/ToRGB".format(h_res, self.width),
                )
            )
            prev_f_m = f_m

    def call(self, inputs, training=None, mask=None):
        x, style = inputs

        y = self.initial_torgb([x, style[:, 0]])

        for idx, (block, torgb) in enumerate(zip(self.synth_blocks, self.torgbs)):
            idx *= 3

            s0 = style[:, idx]
            s1 = style[:, idx + 1]
            s2 = style[:, idx + 2]

            y_h_res = block.out_h_res // 2
            x = block([x, s0, s1])
            y = upsample_height_2d(y, y_h_res, self.width, self.pad0, self.pad1, self.k)
            y = y + torgb([x, s2])

        images_out = y
        return images_out

    def get_config(self):
        config = super(Synthesis, self).get_config()
        config.update(
            {
                "h_resolutions": self.h_resolutions,
                "feat_maps": self.feat_maps,
                "k": self.k,
                "pad0": self.pad0,
                "pad1": self.pad1,
            }
        )
        return config
