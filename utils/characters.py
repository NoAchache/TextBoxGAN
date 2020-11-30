import numpy as np

# coding:utf-8
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&"

# char vector of the OCR used
ASTER_CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Note that '-' is the blank character of the ctc loss, which avoids repeating letters incorrectly.
NUM_CLASSES = len(CHAR_VECTOR)

# TODO: what is char is out of bound?
def encode_label(label):
    return np.array([CHAR_VECTOR.index(x) for x in label])


# TODO: if Aster: i = i-2
def decode_label(encoded_text):
    return "".join([CHAR_VECTOR[i] for i in encoded_text])
