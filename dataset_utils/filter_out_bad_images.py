import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import cv2
from tqdm import tqdm


from config import cfg
from aster_ocr_utils.aster_inferer import AsterInferer
from losses.gan_losses import GeneratorLoss


def filter_out_bad_images():
    loss = GeneratorLoss()
    ocr = AsterInferer()

    with open(os.path.join(cfg.training_dir, "annotations.txt"), "r") as annotations:
        with open(
            os.path.join(cfg.training_dir, "annotations_filtered.txt"), "w"
        ) as annotations_filtered:
            lines = annotations.readlines()

            for i, data in tqdm(enumerate(lines)):

                img_name, label = data.split(",", 1)
                label = label.strip("\n")

                img = cv2.imread(os.path.join(cfg.training_dir, img_name))
                h, w, _ = img.shape

                ocr_img = cv2.resize(
                    img, (cfg.aster_img_dims[0], cfg.aster_img_dims[1])
                )
                ocr_img = ocr_img.astype(np.float32) / 127.5 - 1.0

                ocr_encoded_label = cfg.char_tokenizer.aster.texts_to_sequences([label])
                ocr_padded_label = pad_sequences(
                    ocr_encoded_label, maxlen=cfg.max_chars, value=1, padding="post"
                )[0]

                if len(label) not in range(cfg.min_chars, cfg.max_chars + 1):
                    continue

                ocr_img = tf.expand_dims(tf.constant(ocr_img), 0)
                ocr_padded_label = tf.expand_dims(tf.constant(ocr_padded_label), 0)

                pred = ocr.run(ocr_img)
                l = loss.ocr_loss(pred, ocr_padded_label)
                if l < 10:
                    annotations_filtered.write(data)


if __name__ == "__main__":
    filter_out_bad_images()
