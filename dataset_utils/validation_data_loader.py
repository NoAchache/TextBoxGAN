import os

import numpy as np
import tensorflow as tf

from config import cfg
from utils.utils import string_to_main_int_sequence, string_to_aster_int_sequence

"""
Loads a Tensorflow dataset which is used for validation and testing.

"""


class ValidationDataLoader:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def load_dataset(self, batch_size: int):

        with open(
            os.path.join(cfg.training_text_corpus_dir, self.file_name), "r"
        ) as random_words_file:
            random_words = random_words_file.readlines()

        dataset = (
            tf.data.Dataset.from_tensor_slices(random_words)
            .map(
                lambda data: tf.py_function(
                    func=self._data_getter,
                    inp=[data],
                    Tout=(tf.int32, tf.int32),
                ),
                num_parallel_calls=cfg.num_workers,
            )
            .batch(batch_size, drop_remainder=True)
        )

        return dataset

    def _data_getter(self, data) -> (np.ndarray, np.ndarray):

        word = data.numpy().decode("utf-8")
        word = word.strip("\n")

        input_word_array = string_to_main_int_sequence([word])[0]
        ocr_label_array = string_to_aster_int_sequence([word])[0]

        return input_word_array, ocr_label_array
