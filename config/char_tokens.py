from tensorflow.keras.preprocessing.text import Tokenizer

# Characters available in our model
MAIN_CHAR_VECTOR = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\""
)

# Characters available in the OCR used
ASTER_CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


class CharTokenizer:
    def __init__(self):
        self.main = Tokenizer(char_level=True, lower=False, oov_token="<OOV>")
        self.main.fit_on_texts(MAIN_CHAR_VECTOR)
        self.aster = Tokenizer(char_level=True, lower=False, oov_token="<OOV>")
        self.aster.fit_on_texts(ASTER_CHAR_VECTOR)
