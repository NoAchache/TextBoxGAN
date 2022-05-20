from typing import Tuple

import tensorflow as tf

# Inspired from the pytorch version https://github.com/rosinality/stylegan2-pytorch/blob/master/projector.py

ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth

    allow_memory_growth()

import argparse
import math
import os

import cv2
from tqdm import tqdm

from aster_ocr_utils.aster_inferer import AsterInferer
from config import cfg
from infere import Infere
from models.losses.ocr_losses import softmax_cross_entropy_loss
from models.model_loader import ModelLoader
from projector.lpips_tensorflow import learned_perceptual_metric_model
from utils.loss_tracker import LossTracker
from utils.utils import string_to_aster_int_sequence, string_to_main_int_sequence


class Projector:
    """Projects a text box to find the latent vector responsible for its style."""

    def __init__(self, text_of_the_image):
        self.text_of_the_image = text_of_the_image
        self.image_width = cfg.char_width * len(text_of_the_image)
        self.char_height = cfg.char_height
        perceptual_weights_dir = os.path.join(
            cfg.working_dir, "projector/perceptual_weights"
        )
        self.vgg_ckpt_fn = os.path.join(perceptual_weights_dir, "vgg", "exported")
        self.lin_ckpt_fn = os.path.join(perceptual_weights_dir, "lin", "exported")
        self.perceptual_loss = learned_perceptual_metric_model(
            self.char_height, self.image_width, self.vgg_ckpt_fn, self.lin_ckpt_fn
        )
        self.infere = Infere()

        self.aster_ocr = AsterInferer()
        self.generator = ModelLoader().load_generator(
            is_g_clone=True, ckpt_dir=cfg.ckpt_dir
        )

        self.n_mean_latent = 10000  # number of latents to take the mean from
        self.num_steps = 1000
        self.save_and_log_frequency = 100
        self.lr_rampup = 0.05  # duration of the learning rate warmup
        self.lr_rampdown = 0.25  # duration of the learning rate decay
        self.lr = 0.1
        self.noise_strength_level = 0.05
        self.noise_ramp = 0.75  # duration of the noise level decay
        self.optimizer = tf.keras.optimizers.Adam()
        self.ocr_loss_factor = 0.1

    def _get_lr(self, t: float) -> float:
        """
        Computes a new learning rate.

        Parameters
        ----------
        t: Ratio of the current step over the total number of steps.

        Returns
        -------
        The new learning rate

        """
        lr_ramp = min(1, (1 - t) / self.lr_rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / self.lr_rampup)

        return self.lr * lr_ramp

    def _compute_w_latent(self) -> Tuple["tf.float32", "tf.float32"]:
        """
        Computes the style vector variable to train. This variable is initialized as the mean of self.n_mean_latent
        random style vectors.

        Returns
        -------
        w_latent_std: Standard deviation of the computed mean style vector.
        w_latent_var: Style vector variable to train.

        """
        z_latent = tf.random.normal(shape=[self.n_mean_latent, cfg.z_dim])
        w_latent = self.generator.latent_encoder(z_latent, training=False)[:, 1, :]
        w_latent_mean = tf.reduce_mean(w_latent, axis=0, keepdims=True)
        w_latent_std = (
            tf.reduce_sum((w_latent - w_latent_mean) ** 2) / self.n_mean_latent
        ) ** 0.5

        w_latent_var = tf.Variable(w_latent_mean, name="w_latent_var", trainable=True)
        return w_latent_std, w_latent_var

    def _load_image(self, target_image_path: str, image_width: int) -> tf.float32:
        """
        Load and preprocess the target image.

        Parameters
        ----------
        target_image_path: Path of the image to project.
        image_width: Width of the preprocessed image

        Returns
        -------

        """
        image = cv2.imread(target_image_path)
        image = cv2.resize(image, (image_width, self.char_height))
        return tf.expand_dims(tf.constant(image), 0)

    def main(self, target_image_path: str, output_dir: str) -> None:
        """
        Entry point of the Projector.

        Parameters
        ----------
        target_image_path: Path of the image to project.
        output_dir: Directory on which the output styles and images are saved.

        """
        target_image = self._load_image(target_image_path, self.image_width)
        input_word_array = string_to_main_int_sequence([self.text_of_the_image])
        ocr_label = string_to_aster_int_sequence([self.text_of_the_image])

        w_latent_std, w_latent_var = self._compute_w_latent()

        word_encoded = self.generator.word_encoder(
            input_word_array,
            batch_size=1,
            training=False,
        )

        saved_latents = []
        loss_tracker = LossTracker(["perceptual_loss"])

        for step in tqdm(range(1, self.num_steps + 1)):
            t = step / self.num_steps
            lr = self._get_lr(t)
            self.optimizer.lr.assign(lr)

            noise_strength = (
                w_latent_std
                * self.noise_strength_level
                * max(0, 1 - t / self.noise_ramp) ** 2
            )
            w_latent_noise = tf.random.normal(shape=w_latent_var.shape) * noise_strength

            loss = self._projector_step(
                w_latent_noise,
                w_latent_var,
                ocr_label,
                word_encoded,
                input_word_array,
                target_image,
            )
            loss_tracker.increment_losses({"perceptual_loss": loss})

            if step % self.save_and_log_frequency == 0:
                saved_latents.append(w_latent_var.numpy())
                loss_tracker.print_losses(step)
                self.infere.genererate_chosen_words(
                    [
                        self.text_of_the_image,
                    ],
                    prefix="projected_image" + str(step),
                    output_dir=output_dir,
                    do_sentence=False,
                    w_latents=saved_latents[-1],
                )
                with open(os.path.join(output_dir, "latents.txt"), "w") as file:
                    for latent in saved_latents:
                        file.write(str(latent) + "\n")

    def _get_ocr_loss(
        self, ocr_label: tf.int32, generated_image: tf.float32, input_word: tf.int32
    ) -> tf.float32:
        """
        Computes the softmax crossentropy OCR loss.

        Parameters
        ----------
        ocr_label: Integer sequence obtained from the input word (initially a string) using the ASTER_CHAR_VECTOR.
        generated_image: Text box generated with our model.
        input_word: Integer sequence obtained from the input word (initially a string) using the MAIN_CHAR_VECTOR.

        Returns
        -------
        The OCR loss

        """

        fake_images_ocr_format = self.aster_ocr.convert_inputs(
            generated_image, input_word, blank_label=0
        )

        logits = self.aster_ocr(fake_images_ocr_format)
        return softmax_cross_entropy_loss(logits, ocr_label)

    def get_perceptual_loss(
        self, generated_image: tf.float32, target_image: tf.float32
    ) -> tf.float32:
        """
        Computes the perceptual loss.

        Parameters
        ----------
        generated_image: Text box generated with our model.
        target_image: The text box the projector is trying to extract the style from.

        Returns
        -------
        The perceptual loss

        """
        generated_image = generated_image[:, :, :, : self.image_width]
        generated_image = tf.transpose(generated_image, (0, 2, 3, 1))
        generated_image = (tf.clip_by_value(generated_image, -1.0, 1.0) + 1.0) * 127.5
        return self.perceptual_loss([target_image, generated_image])

    @tf.function()
    def _projector_step(
        self,
        w_latent_noise: tf.float32,
        w_latent_var: tf.Variable,
        ocr_label: tf.int32,
        word_encoded: tf.float32,
        input_word: tf.int32,
        target_image: tf.float32,
    ) -> tf.float32:
        """
        Training step for the projector.

        Parameters
        ----------
        w_latent_noise: Noise applied on w_latent_var.
        w_latent_var: Style vector the projector is training.
        ocr_label: Integer sequence obtained from the input word (initially a string) using the ASTER_CHAR_VECTOR.
        word_encoded: Output of the Word Encoder when inferring input_word.
        input_word: Integer sequence obtained from the input word (initially a string) using the MAIN_CHAR_VECTOR.
        target_image: The text box the projector is trying to extract the style from.

        Returns
        -------
        The resulting loss

        """
        with tf.GradientTape() as tape:
            w_latent_final = tf.tile(
                tf.expand_dims(
                    w_latent_var + w_latent_noise,
                    0,
                ),
                [1, self.generator.n_style, 1],
            )

            generated_image = self.generator.synthesis(
                [word_encoded, w_latent_final], training=False
            )

            ocr_loss = self._get_ocr_loss(ocr_label, generated_image, input_word)
            p_loss = self.get_perceptual_loss(generated_image, target_image)
            loss = p_loss + self.ocr_loss_factor * ocr_loss
        gradients = tape.gradient(loss, [w_latent_var])
        self.optimizer.apply_gradients(zip(gradients, [w_latent_var]))
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_image_path",
        type=str,
        required=True,
        help="path of the image to project",
    )
    parser.add_argument(
        "--text_on_the_image",
        type=str,
        required=True,
        help="text on the image to project",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory on which the images and latents obtained will be saved",
    )
    args = parser.parse_args()

    projector = Projector(args.text_on_the_image)
    projector.main(args.target_image_path, args.output_dir)
