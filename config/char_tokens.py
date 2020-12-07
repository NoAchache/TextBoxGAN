import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


MAIN_CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&"

# char vector of the OCR used
ASTER_CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Note that '-' is the blank character of the ctc loss, which avoids repeating letters incorrectly.
NUM_CLASSES = len(MAIN_CHAR_VECTOR) #TODO: delete

# TODO: what is char is out of bound?
def encode_label(label):
    return np.array([MAIN_CHAR_VECTOR.index(x) for x in label])


# TODO: if Aster: i = i-2
def decode_label(encoded_text):
    return "".join([MAIN_CHAR_VECTOR[i] for i in encoded_text])

class CharTokenizer:
    def __init__(self):
        self.main = Tokenizer(char_level=True, lower=False, oov_token="<OOV>")
        self.main.fit_on_texts(MAIN_CHAR_VECTOR)
        self.aster = Tokenizer(char_level=True, lower=False, oov_token="<OOV>")
        self.aster.fit_on_texts(ASTER_CHAR_VECTOR)



