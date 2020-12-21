import tensorflow as tf

from config import cfg
from models.stylegan2.generator import Generator
from aster_ocr_utils.aster_inferer import AsterInferer


class TensorboardWriter:
    def __init__(self, log_dir: str):
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.num_images_per_log = cfg.num_images_per_log
        self.z_dim = cfg.z_dim
        self.strategy = cfg.strategy
        self.num_images_per_log = min(cfg.batch_size, cfg.num_images_per_log)

    def log_scalars(self, loss_dict: dict, step: int):
        with self.train_summary_writer.as_default():
            for loss_name, metric in loss_dict.items():
                tf.summary.scalar(loss_name, metric.result(), step=step)

    def log_images(
        self,
        input_texts: tf.int32,
        generator: Generator,
        aster_ocr: AsterInferer,
        step: int,
    ):

        test_z = tf.random.normal(
            shape=[self.num_images_per_log, self.z_dim], dtype=tf.dtypes.float32
        )

        if cfg.strategy.num_replicas_in_sync > 1:
            input_texts = input_texts.values  # a list [x_from_dev_a, x_from_dev_b, ...]
            input_texts = tf.concat(input_texts, axis=0)

        input_texts = tf.tile(input_texts[0:1], [self.num_images_per_log, 1])

        batch_concat_imgs, height_concat_imgs = self._gen_samples(
            test_z, input_texts, generator
        )

        # convert to tensor image
        summary_images, ocr_input_images = self._convert_per_replica_image(
            batch_concat_imgs, height_concat_imgs, self.strategy, input_texts, aster_ocr
        )

        text_log = self._get_text_log(input_texts[0:1], ocr_input_images, aster_ocr)

        with self.train_summary_writer.as_default():
            tf.summary.image("images", summary_images, step=step)
            tf.summary.text("texts", text_log, step=step)

    @tf.function
    def _gen_samples(self, z: tf.float32, input_text: tf.int32, generator: Generator):

        # run networks
        fake_images_05 = generator(
            (input_text, z),
            truncation_psi=0.5,
            batch_size=self.num_images_per_log,
            training=False,
        )
        fake_images_07 = generator(
            (input_text, z),
            truncation_psi=0.7,
            batch_size=self.num_images_per_log,
            training=False,
        )

        height_concat_imgs = tf.concat([fake_images_05, fake_images_07], axis=2)
        batch_concat_imgs = tf.concat([fake_images_05, fake_images_07], axis=0)

        return batch_concat_imgs, height_concat_imgs

    @staticmethod
    def _convert_per_replica_image(
        batch_concat_imgs: tf.float32,
        height_concat_imgs: tf.float32,
        strategy: tf.distribute.Strategy,
        input_texts: tf.int32,
        aster_ocr: AsterInferer,
    ):
        summary_images = tf.concat(
            strategy.experimental_local_results(height_concat_imgs), axis=0
        )

        summary_images = tf.transpose(summary_images, perm=[0, 2, 3, 1])
        summary_images = (tf.clip_by_value(summary_images, -1.0, 1.0) + 1.0) * 127.5
        summary_images = tf.cast(summary_images, tf.uint8)

        ocr_input_images = tf.concat(
            strategy.experimental_local_results(batch_concat_imgs), axis=0
        )

        all_input_texts = tf.concat(
            strategy.experimental_local_results(input_texts), axis=0
        )

        ocr_input_images = aster_ocr.convert_inputs(
            ocr_input_images, tf.tile(all_input_texts, [2, 1]), blank_label=0
        )

        return summary_images, ocr_input_images

    def _get_text_log(
        self,
        input_text_code: tf.int32,
        ocr_input_images: tf.float32,
        aster_ocr: AsterInferer,
    ):
        true_text = cfg.char_tokenizer.main.sequences_to_texts(
            input_text_code.numpy() + 1
        )[0]
        logits = aster_ocr(ocr_input_images)
        sequence_length = [logits.shape[1]] * tf.shape(logits)[0].numpy()
        sequences_decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False
        )[0][0]
        sequences_decoded = tf.sparse.to_dense(sequences_decoded).numpy()
        text_list = cfg.char_tokenizer.aster.sequences_to_texts(sequences_decoded)
        return true_text + " / " + " ~~ ".join(text_list)
