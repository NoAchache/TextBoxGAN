ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth

    allow_memory_growth()

import tensorflow as tf
import cv2
import numpy as np
from typing import List
import os

from config import cfg
from models.model_loader import ModelLoader
from utils.loss_tracker import LossTracker
from utils.utils import generator_output_to_rgb, encode_text
from dataset_utils.validation_data_loader import ValidationDataLoader
from validation_step import ValidationStep
from aster_ocr_utils.aster_inferer import AsterInferer

NUM_BOXES = 10
WORDS_TO_GENERATE = ["Hello", "World"]
OUTPUT_PATH = ""
SENTENCE = (
    False  # whether to concat the output boxes in a single image to make a sentence
)


class Infere:
    def __init__(self):
        self.generator = ModelLoader().load_generator(
            is_g_clone=True, ckpt_dir=cfg.ckpt_dir
        )
        self.aster_ocr = AsterInferer()
        self.test_step = ValidationStep(self.generator, self.aster_ocr)

    def genererate_chosen_text(
        self,
        text_list: List[str],
        prefix: str,
        output_path: str,
        sentence: bool,
        w_latents=None,
        truncation_psi=1.0,
    ):
        """
        Generate text boxes for a list of words
        """

        padded_encoded_texts = encode_text(text_list)

        if w_latents is not None:
            word_encoded = self.generator.word_encoder(
                padded_encoded_texts,
                batch_size=len(text_list),
                training=False,
            )
            w_latents = tf.tile(
                tf.expand_dims(w_latents, 0),
                [len(text_list), self.generator.n_style, 1],
            )
            fake_images = self.generator.synthesis(
                [word_encoded, w_latents], training=False
            )
        else:
            fake_images = self.generator(
                [
                    tf.constant(padded_encoded_texts),
                    tf.tile(
                        tf.random.normal(shape=[1, cfg.z_dim]), [len(text_list), 1]
                    ),
                ],
                training=False,
                truncation_psi=truncation_psi,
                batch_size=len(text_list),
            )

        fake_images = generator_output_to_rgb(fake_images)

        if sentence:
            sentence_img = fake_images[0].numpy()[
                :, : cfg.char_width * len(text_list[0])
            ]
            for image, text in zip(fake_images[1:], text_list[1:]):
                new_word_img = image.numpy()[:, : cfg.char_width * len(text)]
                sentence_img = np.concatenate([sentence_img, new_word_img], axis=1)
            cv2.imwrite(
                os.path.join(output_path, prefix + "sentence_image.png"), sentence_img
            )

        else:
            for i, (image, text) in enumerate(zip(fake_images, text_list)):
                cv2.imwrite(
                    os.path.join(output_path, f"{prefix}_{str(i)}_image.png"),
                    image.numpy()[:, : cfg.char_width * len(text)],
                )

    def infere_test_set(self):
        """
        Compute the ocr loss on the test set
        """
        test_loader = ValidationDataLoader("test_corpus.txt")
        test_dataset = test_loader.load_dataset(batch_size=cfg.batch_size)
        loss_tracker = LossTracker(["test_ocr_loss"])

        for step, (input_texts, labels) in enumerate(test_dataset):
            ocr_loss = self.test_step.dist_validation_step(input_texts, labels)
            loss_tracker.increment_losses({"test_ocr_loss": ocr_loss})

        loss_tracker.print_losses(step)


if __name__ == "__main__":
    infere = Infere()
    for i in range(NUM_BOXES):
        infere.genererate_chosen_text(
            WORDS_TO_GENERATE,
            str(i),
            OUTPUT_PATH,
            sentence=SENTENCE,
        )
