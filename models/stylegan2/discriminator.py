import tensorflow as tf

from models.stylegan2.layers.dense import Dense
from models.stylegan2.layers.conv import Conv2D
from models.stylegan2.layers.bias_act import BiasAct
from models.stylegan2.layers.from_rgb import FromRGB
from models.stylegan2.layers.mini_batch_std import MinibatchStd
from config import cfg


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, reduce_height, in_h_res, in_w_res, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.reduce_height = reduce_height
        self.in_h_res = in_h_res
        self.in_w_res = in_w_res

        self.resnet_scale = 1.0 / tf.sqrt(2.0)

        # conv_0
        self.conv_0 = Conv2D(
            in_fmaps=self.n_f0,
            out_fmaps=self.n_f0,
            kernel=3,
            down=False,
            resample_kernel=None,
            gain=self.gain,
            lrmul=self.lrmul,
            name="conv_0",
        )
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_0")

        # conv_1 down
        self.conv_1 = Conv2D(
            in_fmaps=self.n_f0,
            out_fmaps=self.n_f1,
            kernel=3,
            down=True,
            resample_kernel=[1, 3, 3, 1],
            gain=self.gain,
            lrmul=self.lrmul,
            name="conv_1",
            reduce_height=self.reduce_height,
            in_h_res=self.in_h_res,
            in_w_res=self.in_w_res,
        )
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_1")

        # resnet skip
        self.conv_skip = Conv2D(
            in_fmaps=self.n_f0,
            out_fmaps=self.n_f1,
            kernel=1,
            down=True,
            resample_kernel=[1, 3, 3, 1],
            gain=self.gain,
            lrmul=self.lrmul,
            name="skip",
            reduce_height=self.reduce_height,
            in_h_res=self.in_h_res,
            in_w_res=self.in_w_res,
        )

    def call(self, inputs, training=None, mask=None):
        x = inputs
        residual = x

        # conv0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # conv1 down
        x = self.conv_1(x)
        x = self.apply_bias_act_1(x)

        # resnet skip
        residual = self.conv_skip(residual)
        x = (x + residual) * self.resnet_scale

        return x

    def get_config(self):
        config = super(DiscriminatorBlock, self).get_config()
        config.update(
            {
                "n_f0": self.n_f0,
                "n_f1": self.n_f1,
                "gain": self.gain,
                "lrmul": self.lrmul,
                "res": self.res,
                "resnet_scale": self.resnet_scale,
            }
        )
        return config


class DiscriminatorLastBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, res, **kwargs):
        super(DiscriminatorLastBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.res = res

        self.minibatch_std = MinibatchStd(
            group_size=4, num_new_features=1, name="minibatchstd"
        )

        # conv_0
        self.conv_0 = Conv2D(
            in_fmaps=self.n_f0 + 1,
            out_fmaps=self.n_f0,
            kernel=3,
            down=False,
            resample_kernel=None,
            gain=self.gain,
            lrmul=self.lrmul,
            name="conv_0",
        )
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_0")

        # dense_1
        self.dense_1 = Dense(
            self.n_f1, gain=self.gain, lrmul=self.lrmul, name="dense_1"
        )
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act="lrelu", name="bias_1")

    def call(self, x, training=None, mask=None):
        x = self.minibatch_std(x)

        # conv_0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # dense_1
        x = self.dense_1(x)
        x = self.apply_bias_act_1(x)
        return x

    def get_config(self):
        config = super(DiscriminatorLastBlock, self).get_config()
        config.update(
            {
                "n_f0": self.n_f0,
                "n_f1": self.n_f1,
                "gain": self.gain,
                "lrmul": self.lrmul,
                "res": self.res,
            }
        )
        return config


class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        # discriminator's (resolutions and featuremaps) are reversed against generator's
        self.resolutions = cfg.discrim_resolutions
        self.feat_maps = cfg.discrim_feat_maps

        # stack discriminator blocks
        res0, n_f0 = self.resolutions[0], self.feat_maps[0]
        self.initial_fromrgb = FromRGB(
            fmaps=n_f0,
            h_res=res0[0],
            w_res=res0[1],
            name="{:d}x{:d}/FromRGB".format(res0[0], res0[1]),
        )
        self.blocks = list()
        for index, (res0, n_f0) in enumerate(
            zip(self.resolutions[:-1], self.feat_maps[:-1])
        ):
            n_f1 = self.feat_maps[index + 1]
            self.blocks.append(
                DiscriminatorBlock(
                    n_f0=n_f0, n_f1=n_f1, res=res0, name="{:d}x{:d}".format(res0, res0)
                )
            )

        # set last discriminator block
        res = self.resolutions[-1]
        n_f0, n_f1 = self.feat_maps[-2], self.feat_maps[-1]
        self.last_block = DiscriminatorLastBlock(
            n_f0, n_f1, res, name="{:d}x{:d}".format(res, res)
        )

        # set last dense layer
        self.last_dense = Dense(1, gain=1.0, lrmul=1.0, name="last_dense")
        self.last_bias = BiasAct(lrmul=1.0, act="linear", name="last_bias")

    def call(self, inputs, training=None, mask=None):
        images = inputs

        x = self.initial_fromrgb(images)
        for block in self.blocks:
            x = block(x)

        x = self.last_block(x)
        x = self.last_dense(x)
        x = self.last_bias(x)

        scores_out = x
        return scores_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
