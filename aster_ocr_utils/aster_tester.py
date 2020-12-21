import os
import cv2
import numpy as np
import tensorflow as tf


from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg

IMG_FOLDER = "/home/noe/tmp"


def infer_images(aster):
    for img_name in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        ocr_img = cv2.resize(img, (cfg.aster_img_dims[1], cfg.aster_img_dims[0]))
        ocr_img = ocr_img.astype(np.float32) / 127.5 - 1.0
        ocr_img = tf.expand_dims(tf.constant(ocr_img), 0)

        logits = aster(ocr_img)
        sequence_length = [logits.shape[1]] * tf.shape(logits)[0].numpy()
        sequences_decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False
        )[0][0]
        sequences_decoded = tf.sparse.to_dense(sequences_decoded).numpy()
        text = cfg.char_tokenizer.aster.sequences_to_texts(sequences_decoded)[0]
        print(img_path)
        print(text)


if __name__ == "__main__":
    aster = AsterInferer()
    infer_images(aster)
