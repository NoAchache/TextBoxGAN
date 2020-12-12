import tensorflow as tf
from config import cfg

class TensorboardWriter:
    def __init__(self, log_dir):
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.num_images_per_log = cfg.num_images_per_log
        self.z_dim = cfg.z_dim
        self.strategy = cfg.strategy
        self.num_images_per_log = min(cfg.batch_size, cfg.num_images_per_log)


    def log_scalars(self, loss_dict: dict, step):
        with self.train_summary_writer.as_default():
            for loss_name, metric in loss_dict.items():
                tf.summary.scalar(
                        loss_name, metric.result(), step=step
                    )

    def log_images(self, input_texts, generator, aster_ocr, step):

        test_z = tf.random.normal(
                shape=(self.num_images_per_log, self.z_dim),
                dtype=tf.dtypes.float32,
                )

        input_texts = tf.tile(input_texts[0:1], [self.num_images_per_log, 1, 1, 1])

        summary_images = self._dist_gen_samples(test_z, input_texts, generator)

        # convert to tensor image
        summary_images = self._convert_per_replica_image(
                summary_images, self.strategy
                )

        text_log = self._get_text_log(input_texts[0:1], summary_images, aster_ocr)

        with self.train_summary_writer.as_default():
            tf.summary.image("images", summary_images, step=step)
            tf.summary.text("texts", text_log, step=step)

    @tf.function
    def _dist_gen_samples(self, z, input_text, generator):
        return cfg.strategy.experimental_run_v2(
            self._gen_samples, args=(z, input_text, generator)
        )

    def _gen_samples(self, z, input_text, generator):

        # run networks
        fake_images_05 = generator((input_text, z), truncation_psi=0.5, training=False)
        fake_images_07 = generator((input_text, z), truncation_psi=0.7, training=False)

        final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return final_image

    @staticmethod
    def _convert_per_replica_image(nchw_per_replica_images, strategy):
        as_tensor = tf.concat(
            strategy.experimental_local_results(nchw_per_replica_images), axis=0
        )
        as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
        as_tensor = tf.cast(as_tensor, tf.uint8)
        return as_tensor

    def _get_text_log(self, input_text_code, summary_images, aster_ocr):
        true_text = cfg.char_tokenizer.main.sequences_to_texts(input_text_code.numpy()+1)[0]
        logits = aster_ocr(summary_images)
        actual_texts = tf.nn.ctc_greedy_decoder(tf.transpose(logits, [1,0,2]), [logits.shape[1]], merge_repeated=False)[0].values
        return true_text + " / " + " ~~ ".join(actual_texts)




