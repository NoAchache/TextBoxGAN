import tensorflow as tf
from config import cfg

# TODO: print ocr results:
# tf.nn.ctc_greedy_decoder(tf.transpose(forward_logits, [1,0,2]), [forward_logits.shape[1]], merge_repeated=False)[0][0].values
class LogSummary:
    def dist_gen_samples(self, inputs, generator):
        return cfg.strategy.experimental_run_v2(
            self._gen_samples, args=(inputs, generator)
        )

    def _gen_samples(self, inputs, generator):
        test_z = inputs

        # run networks
        fake_images_05 = generator(test_z, truncation_psi=0.5, training=False)
        fake_images_07 = generator(test_z, truncation_psi=0.7, training=False)

        # merge on batch dimension: [n_samples, 3, out_res, 2 * out_res]
        final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return final_image
