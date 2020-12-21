import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import cfg


def mask_text_box(fake_images, input_texts, char_width):
    mask = tf.tile(
        tf.expand_dims(
            tf.expand_dims(
                tf.repeat(
                    tf.where(input_texts == 0, 0.0, 1.0),
                    repeats=tf.tile([char_width], [tf.shape(input_texts)[1]]),
                    axis=1,
                ),
                1,
            ),
            1,
        ),
        [1, fake_images.shape[1], fake_images.shape[2], 1],
    )

    return fake_images * mask


def generator_output_to_rgb(fake_images):
    fake_images = (tf.clip_by_value(fake_images, -1.0, 1.0) + 1.0) * 127.5
    fake_images = tf.transpose(fake_images, perm=[0, 2, 3, 1])
    return tf.cast(fake_images, tf.uint8)


def encode_text(text_of_the_image):
    encoded_texts = cfg.char_tokenizer.main.texts_to_sequences([text_of_the_image])
    # First element is 1 so remove 1 to each element to match embedding shape
    return (
        pad_sequences(encoded_texts, maxlen=cfg.max_chars, value=1, padding="post") - 1
    )
