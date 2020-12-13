import tensorflow as tf

from models.stylegan2.utils import lerp
from models.stylegan2.layers.mapping_block import Mapping
from config import cfg


class LatentEncoder(tf.keras.Model):
    def __init__(self, n_broadcast, **kwargs):

        super(LatentEncoder, self).__init__(**kwargs)

        self.z_dim = cfg.z_dim
        self.style_dim = cfg.style_dim
        self.n_mapping = cfg.n_mapping
        self.w_ema_decay = 0.995
        self.style_mixing_prob = 0.9
        self.n_broadcast = n_broadcast

        self.g_mapping = Mapping(self.style_dim, self.n_mapping, name="g_mapping")
        self.mixing_layer_indices = tf.range(self.n_broadcast, dtype=tf.int32)[
            tf.newaxis, :, tf.newaxis
        ]

        self.broadcast = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[:, tf.newaxis], [1, self.n_broadcast, 1])
        )

    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(
            tf.zeros(shape=[self.style_dim], dtype=tf.dtypes.float32),
            name="w_avg",
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

    def update_moving_average_of_w(self, w_broadcasted):
        # compute average of current w
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)

        # compute moving average of w and update(assign) w_avg
        self.w_avg.assign(lerp(batch_avg, self.w_avg, self.w_ema_decay))
        return

    def style_mixing_regularization(self, latents1, w_broadcasted1):
        # get another w and broadcast it
        latents2 = tf.random.normal(shape=tf.shape(latents1), dtype=tf.dtypes.float32)
        dlatents2 = self.g_mapping(latents2)
        w_broadcasted2 = self.broadcast(dlatents2)

        # find mixing limit index
        # mixing_cutoff_index = tf.cond(
        #     pred=tf.less(tf.random.uniform([], 0.0, 1.0), self.style_mixing_prob),
        #     true_fn=lambda: tf.random.uniform([], 1, self.n_broadcast, dtype=tf.dtypes.int32),
        #     false_fn=lambda: tf.constant(self.n_broadcast, dtype=tf.dtypes.int32))
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform(
                [], 1, self.n_broadcast, dtype=tf.dtypes.int32
            )
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype=tf.dtypes.int32)

        # mix it
        mixed_w_broadcasted = tf.where(
            condition=tf.broadcast_to(
                self.mixing_layer_indices < mixing_cutoff_index,
                tf.shape(w_broadcasted1),
            ),
            x=w_broadcasted1,
            y=w_broadcasted2,
        )
        return mixed_w_broadcasted

    def truncation_trick(self, w_broadcasted, truncation_psi, truncation_cutoff=None):
        ones = tf.ones_like(self.mixing_layer_indices, dtype=tf.float32)
        tpsi = ones * truncation_psi
        if truncation_cutoff is None:
            truncation_coefs = tpsi
        else:
            indices = tf.range(self.n_broadcast)
            truncation_coefs = tf.where(
                condition=tf.less(indices, truncation_cutoff), x=tpsi, y=ones
            )

        truncated_w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)
        return truncated_w_broadcasted

    def call(
        self,
        inputs,
        ret_w_broadcasted=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        training=None,
        mask=None,
    ):
        latents = inputs

        dlatents = self.g_mapping(latents)
        w_broadcasted = self.broadcast(dlatents)

        if training:
            self.update_moving_average_of_w(w_broadcasted)
            w_broadcasted = self.style_mixing_regularization(latents, w_broadcasted)

        if not training:
            w_broadcasted = self.truncation_trick(
                w_broadcasted, truncation_psi, truncation_cutoff
            )

        return w_broadcasted
