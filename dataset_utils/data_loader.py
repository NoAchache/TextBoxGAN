import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import cv2

from config import cfg


def load_dataset(shuffle: bool, epochs: int, batch_size: int) -> tf.data.Dataset:
    dataset = (
        tf.data.Dataset.from_generator(
            data_generator,
            output_types=(tf.float32, tf.int32, tf.int32),
            args=([shuffle]),
        )
        .repeat(epochs)
        .batch(batch_size)
    )
    return dataset


def data_generator(shuffle: bool) -> (np.ndarray, np.ndarray, np.ndarray):
    np.random.seed(cfg.shuffle_seed)

    with open(
        os.path.join(cfg.training_dir, "annotations_filtered.txt"), "r"
    ) as annotations:
        lines = annotations.readlines()
        if shuffle:
            np.random.shuffle(lines)
        for i, data in enumerate(lines):

            img_name, label = data.split(",", 1)
            label = label.strip("\n")
            img = cv2.imread(os.path.join(cfg.training_dir, img_name))
            h, w, _ = img.shape

            img = cv2.resize(img, (cfg.char_width * len(label), cfg.char_height))
            img = img.astype(np.float32) / 127.5 - 1.0

            padding_length = (cfg.max_chars - len(label)) * cfg.char_width
            padded_img = cv2.copyMakeBorder(
                src=img,
                top=0,
                bottom=0,
                left=0,
                right=padding_length,
                borderType=cv2.BORDER_CONSTANT,
            )

            padded_img = np.transpose(padded_img, (2, 0, 1))  # H,W,C to C,H,W
            main_encoded_label = cfg.char_tokenizer.main.texts_to_sequences([label])
            main_padded_label = (
                pad_sequences(
                    main_encoded_label, maxlen=cfg.max_chars, value=1, padding="post"
                )[0]
                - 1.0
            )
            # TODO:ensure that works and explain why with comment
            ocr_encoded_label = cfg.char_tokenizer.aster.texts_to_sequences([label])
            ocr_padded_label = pad_sequences(
                ocr_encoded_label, maxlen=cfg.max_chars, value=1, padding="post"
            )[0]

            yield padded_img, main_padded_label, ocr_padded_label
