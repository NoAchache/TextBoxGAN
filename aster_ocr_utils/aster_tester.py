import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg

IMAGES_DIR = ""


def infer_images(aster, images_dir):
    """
    Infer images in the Aster OCR to see how it performs on them.

    Parameters
    ----------
    aster: pre-trained OCR.
    images_dir: Directory containing the images to infer

    """
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        image = np.array(Image.open(image_path))
        ocr_image = tf.cast(tf.convert_to_tensor(image), tf.float32) / 127.5 - 1.0
        ocr_image = tf.compat.v1.image.resize(ocr_image, cfg.aster_image_dims)[
            tf.newaxis
        ]

        logits = aster(ocr_image)
        sequence_length = [logits.shape[1]]
        sequences_decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False
        )[0][0]
        sequences_decoded = tf.sparse.to_dense(sequences_decoded).numpy()
        word = cfg.char_tokenizer.aster.sequences_to_texts(sequences_decoded)[0]
        print(image_path)
        print(word)


if __name__ == "__main__":
    aster_model = AsterInferer()
    infer_images(aster_model, IMAGES_DIR)
