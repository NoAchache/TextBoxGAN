import numpy as np

# coding:utf-8
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&"
# Note that '-' is the blank character of the ctc loss, which avoids repeating letters incorrectly.
NUM_CLASSES = len(CHAR_VECTOR)


def encode_label(label):
    return np.array([CHAR_VECTOR.index(x) for x in label])


def decode_label(encoded_text):
    return "".join([CHAR_VECTOR[i] for i in encoded_text])
