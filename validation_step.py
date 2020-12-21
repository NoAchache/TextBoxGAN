import tensorflow as tf

from config import cfg
from utils.utils import mask_text_box
from models.losses.ocr_losses import softmax_cross_entropy_loss
from models.stylegan2.generator import Generator
from aster_ocr_utils.aster_inferer import AsterInferer


class ValidationStep:
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
        input_texts: tf.int32,
        labels: tf.int32,
    ):

        ocr_loss = cfg.strategy.experimental_run_v2(
            fn=self.validation_step,
            args=(
                input_texts,
                labels,
            ),
        )
        mean_ocr_loss = cfg.strategy.reduce(
            tf.distribute.ReduceOp.SUM, ocr_loss, axis=None
        )

        return mean_ocr_loss

    def validation_step(
        self,
        input_texts: tf.int32,
        labels: tf.int32,
    ):
        z = tf.random.normal(
            shape=[self.batch_size_per_gpu, self.z_dim],
            dtype=tf.dtypes.float32,
        )

        fake_images = self.generator([input_texts, z], training=False)

        fake_images = mask_text_box(fake_images, input_texts, self.char_width)

        ocr_input_image = self.aster_ocr.convert_inputs(
            fake_images, labels, blank_label=1
        )
        logits = self.aster_ocr(ocr_input_image)

        return softmax_cross_entropy_loss(logits, labels)
