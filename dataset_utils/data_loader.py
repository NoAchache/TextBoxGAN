from config import cfg
from utils import encode_label

import tensorflow as tf
import os
import numpy as np
import cv2

def load_dataset(
    shuffle: bool, epochs: int, batch_size: int
) -> tf.data.Dataset:
    dataset = (
        tf.data.Dataset.from_generator(
            data_generator,
            output_types=(tf.float32, tf.float32, tf.int8),
            args=([shuffle]),
        )
        .repeat(epochs)
        .batch(batch_size)
    )
    return dataset

def data_generator(shuffle: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    np.random.seed(cfg.shuffle_seed)

    with open(
        os.path.join(cfg.training_dir, "annotations.txt"), "r"
    ) as annotations:
        lines = annotations.readlines()
        if shuffle:
            np.random.shuffle(lines)
        for i, data in enumerate(lines):

            img_name, label = data.split(",", 1)
            label = label.strip("\n")
            if len(label) not in range(cfg.min_chars, cfg.max_chars + 1):
                continue
            img = cv2.imread(os.path.join(cfg.training_dir, img_name))
            h, w, _ = img.shape

            main_img = cv2.resize(img, (cfg.char_width * len(label), cfg.char_height))
            ocr_img = cv2.resize(img, cfg.aster_img_dims)

            main_img = main_img.astype(np.float32) / 127.5 - 1.0
            ocr_img = ocr_img.astype(np.float32) / 127.5 - 1.0

            padding_length = (cfg.max_chars - len(label)) * cfg.char_width
            padded_img = cv2.copyMakeBorder(
                src=main_img,
                top=0,
                bottom=0,
                left=0,
                right=padding_length,
                borderType=cv2.BORDER_CONSTANT,
            )

            padded_img = np.transpose(padded_img, (2, 0, 1))  # H,W,C to C,H,W
            encoded_label = encode_label(label)
            padded_label = np.concatenate(
                (encoded_label, np.array([-1] * (cfg.max_chars - len(label))))
            )

            yield padded_img, ocr_img,  padded_label
