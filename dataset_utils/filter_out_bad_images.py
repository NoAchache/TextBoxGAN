import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from models.losses.ocr_losses import softmax_cross_entropy_loss
from utils.utils import string_to_aster_int_sequence

OCR_LOSS_THRESHOLD = 15


def filter_out_bad_images() -> None:
    """Filters out the images of the text box dataset for which the OCR loss is below the OCR_LOSS_THRESHOLD"""

    print("Filtering out bad images")
    aster_ocr = AsterInferer()

    with open(
        os.path.join(cfg.training_text_boxes_dir, "annotations.txt"), "r"
    ) as annotations:
        with open(
            os.path.join(cfg.training_text_boxes_dir, "annotations_filtered.txt"), "w"
        ) as annotations_filtered:
            lines = annotations.readlines()

            for i, data in tqdm(enumerate(lines)):
                image_name, word = data.split(",", 1)
                word = word.strip("\n")

                if len(word) > cfg.max_char_number or len(word) == 0:
                    continue

                image = cv2.imread(
                    os.path.join(cfg.training_text_boxes_dir, image_name)
                )
                h, w, _ = image.shape

                image = cv2.resize(
                    image, (cfg.aster_image_dims[1], cfg.aster_image_dims[0])
                )
                image = image.astype(np.float32) / 127.5 - 1.0
                image = tf.expand_dims(tf.constant(image), 0)

                ocr_label_array = tf.constant(string_to_aster_int_sequence([word]))

                prediction = aster_ocr(image)

                loss = (
                    softmax_cross_entropy_loss(prediction, ocr_label_array)
                    * cfg.batch_size
                )
                if loss < OCR_LOSS_THRESHOLD:
                    annotations_filtered.write(data)


if __name__ == "__main__":
    filter_out_bad_images()
