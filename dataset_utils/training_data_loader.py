import os
from random import random
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

from config import cfg
from utils.utils import string_to_main_int_sequence, string_to_aster_int_sequence


class TrainingDataLoader:
    """ Loads a Tensorflow dataset which is used for training. """

    def __init__(self):
        self.return_ocr_image = cfg.ocr_loss_type == "mse"
        self.use_corpus_word = cfg.ocr_loss_type == "softmax_crossentropy"
        with open(
            os.path.join(cfg.training_text_corpus_dir, "train_corpus.txt"), "r"
        ) as corpus_words_file:
            self.corpus_words = corpus_words_file.readlines()
        self.corpus_words_generator = iter(self.corpus_words)
        self.corpus_word_ratio = 0.25

    def load_dataset(self, batch_size: int) -> tf.data.Dataset:
        with open(
            os.path.join(cfg.training_text_boxes_dir, "annotations_filtered.txt"), "r"
        ) as annotations_file:
            annotations_lines = annotations_file.readlines()
            print(len(annotations_lines))

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
                buffer_size=len(annotations_lines)
                if cfg.buffer_size == -1
                else cfg.buffer_size,
                seed=cfg.shuffle_seed,
                reshuffle_each_iteration=True,
            )
            .batch(batch_size, drop_remainder=True)
        )

        return dataset

    def _data_getter(
        self, data
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        data = data.numpy().decode("utf-8")
        image_name, word = data.split(",", 1)
        word = word.strip("\n")
        image = cv2.imread(os.path.join(cfg.training_text_boxes_dir, image_name))

        main_image = cv2.resize(image, (cfg.char_width * len(word), cfg.char_height))
        main_image = main_image.astype(np.float32) / 127.5 - 1.0

        if self.return_ocr_image:
            ocr_image = cv2.resize(
                image, (cfg.aster_image_dims[1], cfg.aster_image_dims[0])
            )
            ocr_image = ocr_image.astype(np.float32) / 127.5 - 1.0
        else:
            ocr_image = 0.0

        padding_length = (cfg.max_char_number - len(word)) * cfg.char_width
        padded_image = cv2.copyMakeBorder(
            src=main_image,
            top=0,
            bottom=0,
            left=0,
            right=padding_length,
            borderType=cv2.BORDER_CONSTANT,
        )

        padded_image = np.transpose(padded_image, (2, 0, 1))  # H,W,C to C,H,W

        if self.use_corpus_word and random() > 1 - self.corpus_word_ratio:
            word = next(self.corpus_words_generator, None)
            if word is None:
                self.corpus_words_generator = iter(self.corpus_words)
                word = next(self.corpus_words_generator)

        input_word_array = string_to_main_int_sequence([word])[0]
        ocr_label_array = string_to_aster_int_sequence([word])[0]

        return padded_image, ocr_image, input_word_array, ocr_label_array
