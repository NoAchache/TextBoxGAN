from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import cfg


def mask_text_box(
    fake_images: tf.float32, input_words: tf.int32, char_width: int
) -> tf.float32:
    """
    Masks the text boxes outputted by the generator, in the cases where the length of the word is less than
    cfg.max_char_number. Since each character is supposed to take 1/cfg.max_char_number of the width of the text box,
    this function masks the extra width.

    Parameters
    ----------
    fake_images: Text boxes generated with our model.
    input_words: Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.
    char_width: Width of a single character.

    Returns
    -------
    Masked fake_images

    """
    mask = tf.tile(
        tf.expand_dims(
            tf.expand_dims(
                tf.repeat(
                    tf.where(input_words == 0, 0.0, 1.0),
                    repeats=tf.tile([char_width], [tf.shape(input_words)[1]]),
                    axis=1,
                ),
                1,
            ),
            1,
        ),
        [1, fake_images.shape[1], fake_images.shape[2], 1],
    )

    return fake_images * mask


def generator_output_to_uint8(fake_images: tf.float32) -> tf.uint8:
    """
    Converts the output of the generator to uint8 RGB images.

    Parameters
    ----------
    fake_images: Text boxes generated with our model.

    Returns
    -------
    Generated text boxes in a uint8 RGB format.

    """
    fake_images = (tf.clip_by_value(fake_images, -1.0, 1.0) + 1.0) * 127.5
    fake_images = tf.transpose(fake_images, perm=[0, 2, 3, 1])
    return tf.cast(fake_images, tf.uint8)


def string_to_main_int_sequence(words_list: List[str]) -> np.ndarray:
    """
    Converts input strings to integer sequences using the main character vector, and pad them if their length are less
    than cfg.max_char_number.

    Parameters
    ----------
    words_list: List of words to generate

    Returns
    -------
    Integer sequences obtained from the input words (initially strings) using the MAIN_CHAR_VECTOR.

    """
    int_sequence = cfg.char_tokenizer.main.texts_to_sequences(words_list)
    # First element is 1 so remove 1 to each element to match embedding shape
    return (
        pad_sequences(int_sequence, maxlen=cfg.max_char_number, value=1, padding="post")
        - 1
    )


def string_to_aster_int_sequence(words_list: List[str]) -> np.ndarray:
    """
    Converts input strings to integer sequences using aster's character vector, and pad them if their length are less
    than cfg.max_char_number.

    Parameters
    ----------
    words_list: List of words to generate

    Returns
    -------
    Integer sequences obtained from the input words (initially strings) using the ASTER_CHAR_VECTOR.

    """
    int_sequence = cfg.char_tokenizer.aster.texts_to_sequences(words_list)
    return pad_sequences(
        int_sequence, maxlen=cfg.max_char_number, value=1, padding="post"
    )
