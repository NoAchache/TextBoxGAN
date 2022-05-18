import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from models.losses.ocr_losses import softmax_cross_entropy_loss
from models.stylegan2.generator import Generator
from utils.utils import mask_text_box


class ValidationStep:
    """Inference class to compute the loss on images generated from input text words in order to validate the model."""

    def __init__(
        self,
        generator: Generator,
        aster_ocr: AsterInferer,
    ):
        self.batch_size_per_gpu = cfg.batch_size_per_gpu
        self.z_dim = cfg.z_dim
        self.generator = generator
        self.char_width = cfg.char_width
        self.aster_ocr = aster_ocr

    @tf.function
    def dist_validation_step(
        self,
        input_words: tf.int32,
        ocr_labels: tf.int32,
    ):
        """
        Entry point of the class. Distributes the validation step on the available GPUs.

        Parameters
        ----------
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
        ocr_labels: Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.

        Returns
        -------
        Mean OCR loss obtained for the text boxes generated from the input_words.

        """

        ocr_loss = cfg.strategy.run(
            fn=self._validation_step,
            args=(
                input_words,
                ocr_labels,
            ),
        )
        mean_ocr_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, ocr_loss, axis=None
        )

        return mean_ocr_loss

    def _validation_step(
        self,
        input_words: tf.int32,
        ocr_labels: tf.int32,
    ):
        """
        Generates text boxes from the input_words and compute their OCR loss.

        Parameters
        ----------
        input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR
        ocr_labels: Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.

        Returns
        -------
        OCR loss obtained for the text boxes generated from the input_words.


        """
        z = tf.random.normal(
            shape=[self.batch_size_per_gpu, self.z_dim],
            dtype=tf.dtypes.float32,
        )

        fake_images = self.generator([input_words, z], training=False)

        fake_images = mask_text_box(fake_images, input_words, self.char_width)

        ocr_input_image = self.aster_ocr.convert_inputs(
            fake_images, ocr_labels, blank_label=1
        )
        logits = self.aster_ocr(ocr_input_image)

        return softmax_cross_entropy_loss(logits, ocr_labels)
