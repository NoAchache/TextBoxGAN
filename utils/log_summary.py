import tensorflow as tf
from utils import cfg


class LogSummary:
    def dist_gen_samples(self, inputs, generator):
        return cfg.strategy.experimental_run_v2(
            self._gen_samples, args=(inputs, generator)
        )

    def _gen_samples(self, inputs, generator):
        test_z, test_labels = inputs

        # run networks
        fake_images_05 = generator(
            [test_z, test_labels], truncation_psi=0.5, training=False
        )
        fake_images_07 = generator(
            [test_z, test_labels], truncation_psi=0.7, training=False
        )

        # merge on batch dimension: [n_samples, 3, out_res, 2 * out_res]
        final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return final_image
