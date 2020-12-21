import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import cv2
from random import random

from config import cfg
from utils.utils import encode_text


class TrainingDataLoader:
    def __init__(self):
        self.return_ocr_image = cfg.ocr_loss == "mse"
        self.use_random_word = cfg.ocr_loss == "softmax_crossentropy"
        with open(
            os.path.join(cfg.training_text_corus_dir, "train_corpus.txt"), "r"
        ) as random_words_file:
            self.random_words = random_words_file.readlines()
        self.random_words_generator = iter(self.random_words)

    def load_dataset(self, batch_size: int):
        with open(
            os.path.join(cfg.training_text_boxes_dir, "annotations_filtered.txt"), "r"
        ) as annotations_file:
            annotations_lines = annotations_file.readlines()

        dataset = (
            tf.data.Dataset.from_tensor_slices(annotations_lines)
            .map(
                lambda data: tf.py_function(
                    func=self._data_getter,
                    inp=[data],
                    Tout=(tf.float32, tf.float32, tf.int32, tf.int32),
                ),
                num_parallel_calls=cfg.num_workers,
            )
            .repeat()
            .shuffle(
                buffer_size=len(annotations_lines),
                seed=cfg.shuffle_seed,
                reshuffle_each_iteration=True,
            )
            .batch(batch_size, drop_remainder=True)
        )

        return dataset

    def _data_getter(self, data) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        data = data.numpy().decode("utf-8")
        img_name, label = data.split(",", 1)
        label = label.strip("\n")
        img = cv2.imread(os.path.join(cfg.training_text_boxes_dir, img_name))

        main_img = cv2.resize(img, (cfg.char_width * len(label), cfg.char_height))
        main_img = main_img.astype(np.float32) / 127.5 - 1.0

        if self.return_ocr_image:
            ocr_img = cv2.resize(img, (cfg.aster_img_dims[1], cfg.aster_img_dims[0]))
            ocr_img = ocr_img.astype(np.float32) / 127.5 - 1.0
        else:
            ocr_img = 0.0

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

        if self.use_random_word and random() > 0.5:
            label = next(self.random_words_generator, None)
            if label is None:
                self.random_words_generator = iter(self.random_words)
                label = next(self.random_words_generator)

        main_padded_label = encode_text([label])

        ocr_encoded_label = cfg.char_tokenizer.aster.texts_to_sequences([label])
        ocr_padded_label = pad_sequences(
            ocr_encoded_label, maxlen=cfg.max_chars, value=1, padding="post"
        )[0]

        return padded_img, ocr_img, main_padded_label, ocr_padded_label
