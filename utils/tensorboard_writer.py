import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from models.stylegan2.generator import Generator
from utils.utils import generator_output_to_uint8


class TensorboardWriter:
    def __init__(self, log_dir: str):
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.num_images_per_log = cfg.num_images_per_log
        self.z_dim = cfg.z_dim
        self.strategy = cfg.strategy
        self.num_images_per_log = min(cfg.batch_size, cfg.num_images_per_log)

    def log_scalars(self, loss_dict: dict, step: int):
        """
        Save the losses.

        Parameters
        ----------
        loss_dict: {name of the loss : value of the loss}
        step: Current training step

        """
        with self.train_summary_writer.as_default():
            for loss_name, metric in loss_dict.items():
                tf.summary.scalar(loss_name, metric.result(), step=step)

    def log_images(
        self,
        input_words: tf.int32,
        generator: Generator,
        aster_ocr: AsterInferer,
        step: int,
    ):
        """
        Generates text boxes and saves them.

        Parameters
        ----------
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        generator: Generator used for inference (moving average of the trained Generator).
        aster_ocr: Pre-trained OCR.
        step: Current training step.

        """
        test_z = tf.random.normal(
            shape=[self.num_images_per_log, self.z_dim], dtype=tf.dtypes.float32
        )

        if cfg.strategy.num_replicas_in_sync > 1:
            input_words = input_words.values  # a list [x_from_dev_a, x_from_dev_b, ...]
            input_words = tf.concat(input_words, axis=0)

        input_words = tf.tile(input_words[0:1], [self.num_images_per_log, 1])

        batch_concat_images, height_concat_images = self._gen_samples(
            test_z, input_words, generator
        )

        (
            batch_concat_images,
            height_concat_images,
            input_words,
        ) = self._convert_per_replica_tensor(
            self.strategy,
            batch_concat_images,
            height_concat_images,
            input_words,
        )

        ocr_images = aster_ocr.convert_inputs(
            batch_concat_images, tf.tile(input_words, [2, 1]), blank_label=0
        )

        text_log = self._get_text_log(input_words[0:1], ocr_images, aster_ocr)
        summary_images = generator_output_to_uint8(height_concat_images)

        with self.train_summary_writer.as_default():
            tf.summary.image("images", summary_images, step=step)
            tf.summary.text("words", text_log, step=step)

    @tf.function
    def _gen_samples(self, z: tf.float32, input_words: tf.int32, generator: Generator):
        """

        Parameters
        ----------
        z: Normally distributed random vector used to generate the style vector.
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        generator: Generator used for inference (moving average of the trained Generator).

        Returns
        -------
        batch_concat_images: Generated text boxes concatenated in the batch dimension
        height_concat_images: Generated text boxes concatenated in the height dimension

        """

        # run networks
        fake_images_05 = generator(
            (input_words, z),
            truncation_psi=0.5,
            batch_size=self.num_images_per_log,
            training=False,
        )
        fake_images_07 = generator(
            (input_words, z),
            truncation_psi=0.7,
            batch_size=self.num_images_per_log,
            training=False,
        )

        height_concat_images = tf.concat([fake_images_05, fake_images_07], axis=2)
        batch_concat_images = tf.concat([fake_images_05, fake_images_07], axis=0)

        return batch_concat_images, height_concat_images

    @staticmethod
    def _convert_per_replica_tensor(
        strategy: tf.distribute.Strategy, *per_replica_tensors
    ):
        """
        Concat the tensors distributed over the different GPU replicas.

        Parameters
        ----------
        strategy: Strategy used to distribute the GPUs.
        per_replica_tensors: tensor distributed over the GPU replicas.

        Returns
        -------
        Concatenated tensors

        """
        concatenated_tensors = []

        for per_replica_tensor in per_replica_tensors:

            concatenated_tensors.append(
                tf.concat(
                    strategy.experimental_local_results(per_replica_tensor), axis=0
                )
            )

        return concatenated_tensors

    def _get_text_log(
        self,
        input_word_array: tf.int32,
        ocr_images: tf.float32,
        aster_ocr: AsterInferer,
    ):
        """
        Reads the text in the generated text boxes using the OCR and converts the integer arrays in strings.

        Parameters
        ----------
        input_word_array: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        ocr_images: Generated text boxes processed to be in the format of the OCR model's inputs.
        aster_ocr: Pre-trained OCR.

        Returns
        -------
        A string containing the input word and the words the OCR read in the generated text boxes.

        """
        true_text = cfg.char_tokenizer.main.sequences_to_texts(
            input_word_array.numpy() + 1
        )[0]
        logits = aster_ocr(ocr_images)
        sequence_length = [logits.shape[1]] * tf.shape(logits)[0].numpy()
        sequences_decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False
        )[0][0]
        sequences_decoded = tf.sparse.to_dense(sequences_decoded).numpy()
        text_list = cfg.char_tokenizer.aster.sequences_to_texts(sequences_decoded)
        return true_text + " / " + " ~~ ".join(text_list)
