import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np

from config import cfg
from utils.utils import encode_text


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

        main_padded_label = encode_text([word])

        ocr_encoded_label = cfg.char_tokenizer.aster.texts_to_sequences([word])
        ocr_padded_label = pad_sequences(
            ocr_encoded_label, maxlen=cfg.max_chars, value=1, padding="post"
        )[0]

        return main_padded_label, ocr_padded_label
