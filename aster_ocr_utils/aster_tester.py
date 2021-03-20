import os

import cv2
import numpy as np
import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg

IMAGES_DIR = ""


def infer_images(aster, images_dir):
    """
    Infere images in the Aster OCR to see how it performs on them.

    Parameters
    ----------
    aster: pre-trained OCR.
    images_dir: Directory containing the images to infere

    """
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(image_name, image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        ocr_image = cv2.resize(
            image, (cfg.aster_image_dims[1], cfg.aster_image_dims[0])
        )
        ocr_image = ocr_image.astype(np.float32) / 127.5 - 1.0
        ocr_image = tf.expand_dims(tf.constant(ocr_image), 0)

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
    aster = AsterInferer()
    infer_images(aster, IMAGES_DIR)
