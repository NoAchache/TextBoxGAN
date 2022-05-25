ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth

    allow_memory_growth()

import argparse
import os
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from dataset_utils.validation_data_loader import ValidationDataLoader
from models.model_loader import ModelLoader
from utils.loss_tracker import LossTracker
from utils.utils import generator_output_to_uint8, string_to_main_int_sequence
from validation_step import ValidationStep


class Infer:
    """Infer the trained model"""

    def __init__(self):
        self.generator = ModelLoader().load_generator(
            is_g_clone=True, ckpt_dir=cfg.ckpt_dir
        )
        self.aster_ocr = AsterInferer()
        self.test_step = ValidationStep(self.generator, self.aster_ocr)
        self.strategy = cfg.strategy

    def genererate_chosen_words(
        self,
        words_list: List[str],
        prefix: str,
        output_dir: str,
        do_sentence: bool,
        w_latents=None,
        truncation_psi=1.0,
    ):
        """
        Generate text boxes from a list of words.

        Parameters
        ----------
        words_list: List of words to generate.
        prefix: Prefix for the names of each output files.
        output_dir: Directory of the output images.
        do_sentence: Whether to concatenate the output words to simulate a sentence.
        w_latents: Style vector for the generated text boxes (e.g. a style vector obtained from the Projector).
        truncation_psi: Truncation threshold (Used in the Latent Encoder).

        """
        padded_encoded_words = string_to_main_int_sequence(words_list)

        if w_latents is not None:
            word_encoded = self.generator.word_encoder(
                padded_encoded_words,
                batch_size=len(words_list),
            )
            w_latents = tf.tile(
                tf.expand_dims(w_latents, 0),
                [len(words_list), self.generator.n_style, 1],
            )
            fake_images = self.generator.synthesis([word_encoded, w_latents])
        else:
            fake_images = self.generator(
                [
                    tf.constant(padded_encoded_words),
                    tf.tile(
                        tf.random.normal(shape=[1, cfg.z_dim]), [len(words_list), 1]
                    ),
                ],
                truncation_psi=truncation_psi,
                batch_size=len(words_list),
            )

        fake_images = generator_output_to_uint8(fake_images)

        if do_sentence:
            sentence_image = fake_images[0].numpy()[
                :, : cfg.char_width * len(words_list[0])
            ]
            for image, word in zip(fake_images[1:], words_list[1:]):
                new_word_image = image.numpy()[:, : cfg.char_width * len(word)]
                sentence_image = np.concatenate(
                    [sentence_image, new_word_image], axis=1
                )
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}_sentence_image.png"), sentence_image
            )

        else:
            for image, word in zip(fake_images, words_list):
                cv2.imwrite(
                    os.path.join(output_dir, f"{prefix}_{word}_image.png"),
                    image.numpy()[:, : cfg.char_width * len(word)],
                )

    def infer_test_set(self, num_test_set_runs):
        """
        Computes the OCR loss for the test set. Takes an average over several runs to mitigate the bias due to the fact
        random vectors are used.

        Parameters
        ----------
        num_test_set_runs: number of times the Test set is inferred.

        """
        test_loader = ValidationDataLoader("test_corpus.txt")
        test_dataset = test_loader.load_dataset(batch_size=cfg.batch_size)
        test_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        global_loss_tracker = LossTracker(["test_ocr_loss"])

        for _ in range(num_test_set_runs):
            loss_tracker = LossTracker(["test_ocr_loss"])

            for step, (input_words, ocr_labels) in enumerate(test_dataset):
                ocr_loss = self.test_step.dist_validation_step(input_words, ocr_labels)
                loss_tracker.increment_losses({"test_ocr_loss": ocr_loss})

            loss_tracker.print_losses(step)
            test_average_ocr_loss = (
                loss_tracker.losses["test_ocr_loss"].result().numpy()
            )
            global_loss_tracker.increment_losses(
                {"test_ocr_loss": test_average_ocr_loss}
            )

        print("_________AVERAGE TEST LOSS___________")
        global_loss_tracker.print_losses(step=NUM_TEST_SET_RUNS)


if __name__ == "__main__":

    # infer test set
    NUM_TEST_SET_RUNS = 100

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer_type",
        type=str,
        required=True,
        help="possible values are 'test_set' or 'chosen_words",
    )

    # If infer_type is 'test_set'
    parser.add_argument(
        "--num_test_set_run",
        type=int,
        default=100,
        help="amount of runs of the test set",
    )

    # If infer_type is 'chosen_words'

    parser.add_argument(
        "--num_inferences",
        type=int,
        default=20,
        help="number of times the input words are inferred",
    )

    parser.add_argument(
        "--words_to_generate",
        nargs="+",
        type=str,
        help="words to generate",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory on which the generated images will be saved",
    )

    parser.add_argument(
        "--sentence",
        default=False,
        action="store_true",
        help=(
            "whether the generated words should be concatenated in a single box to"
            " simulate a sentence"
        ),
    )
    args = parser.parse_args()

    infer = Infer()

    if args.infer_type == "chosen_words":
        for i in range(args.num_inferences):
            infer.genererate_chosen_words(
                args.words_to_generate,
                str(i),
                args.output_dir,
                do_sentence=args.sentence,
            )
    elif args.infer_type == "test_set":
        infer.infer_test_set(args.num_test_set_run)
    else:
        print(
            f"infer_type should be 'chosen_words' or 'test_set', not {args.infer_type}"
        )
