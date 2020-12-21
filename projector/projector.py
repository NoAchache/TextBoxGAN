# Inspired from the pytorch version https://github.com/rosinality/stylegan2-pytorch/blob/master/projector.py
"""
Projects a text box to find the latent vector responsible for its style.
"""
ALLOW_MEMORY_GROWTH = True

if ALLOW_MEMORY_GROWTH:
    # this needs to be instantiated before any file using tf
    from allow_memory_growth import allow_memory_growth

    allow_memory_growth()

import math
import os
import tensorflow as tf
from tqdm import tqdm
import cv2

from config import cfg
from utils.utils import encode_text
from models.model_loader import ModelLoader
from projector.lpips_tensorflow import learned_perceptual_metric_model
from utils.loss_tracker import LossTracker
from infere import Infere


TARGET_IMG_PATH = ""
TEXT_ON_IMG = ""
OUTPUT_PATH = ""


class Projector:
    def __init__(self):
        self.generator = ModelLoader().load_generator(
            is_g_clone=True, ckpt_dir=cfg.ckpt_dir
        )
        self.output_dir = os.path.join(cfg.working_dir, "projector/projector_latents")
        perceptual_weights_dir = os.path.join(
            cfg.working_dir, "projector/perceptual_weights"
        )
        self.vgg_ckpt_fn = os.path.join(perceptual_weights_dir, "vgg", "exported")
        self.lin_ckpt_fn = os.path.join(perceptual_weights_dir, "lin", "exported")

        self.n_mean_latent = 10000  # number of latents to take the mean from
        self.num_steps = 1000
        self.save_and_log_frequency = 100
        self.lr_rampup = 0.05  # duration of the learning rate warmup
        self.lr_rampdown = 0.25  # duration of the learning rate decay
        self.lr = 0.1
        self.noise_strength_level = 0.05
        self.noise_ramp = 0.75  # duration of the noise level decay
        self.optimizer = tf.keras.optimizers.Adam()
        self.char_height = cfg.char_height
        self.char_width = cfg.char_width

    def get_lr(self, t: float):
        lr_ramp = min(1, (1 - t) / self.lr_rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / self.lr_rampup)

        return self.lr * lr_ramp

    def compute_w_latent(self):
        z_latent = tf.random.normal(shape=[self.n_mean_latent, cfg.z_dim])
        w_latent = self.generator.latent_encoder(z_latent, training=False)[:, 1, :]
        w_latent_mean = tf.reduce_mean(w_latent, axis=0, keepdims=True)
        w_latent_std = (
            tf.reduce_sum((w_latent - w_latent_mean) ** 2) / self.n_mean_latent
        ) ** 0.5

        w_latent_var = tf.Variable(w_latent_mean, name="w_latent_var", trainable=True)
        return w_latent_mean, w_latent_std, w_latent_var

    def load_image(self, target_image_path: str, image_width: int):
        img = cv2.imread(target_image_path)
        img = cv2.resize(img, (image_width, self.char_height))
        return tf.expand_dims(tf.constant(img), 0)

    def main(
        self,
        target_image_path: str,
        text_of_the_image: str,
        output_dir: str,
    ):
        image_width = self.char_width * len(text_of_the_image)
        target_image = self.load_image(target_image_path, image_width)
        perceptual_loss = learned_perceptual_metric_model(
            self.char_height, image_width, self.vgg_ckpt_fn, self.lin_ckpt_fn
        )

        padded_encoded_text = encode_text([text_of_the_image])

        w_latent_mean, w_latent_std, w_latent_var = self.compute_w_latent()

        word_encoded = self.generator.word_encoder(
            padded_encoded_text,
            batch_size=1,
            training=False,
        )

        saved_latents = []
        loss_tracker = LossTracker(["perceptual_loss"])

        for step in tqdm(range(self.num_steps)):
            with tf.GradientTape() as tape:
                tape.watch(w_latent_var)
                t = step / self.num_steps
                lr = self.get_lr(t)
                self.optimizer.lr.assign(lr)

                noise_strength = (
                    w_latent_std
                    * self.noise_strength_level
                    * max(0, 1 - t / self.noise_ramp) ** 2
                )
                w_latent_noise = (
                    tf.random.normal(shape=w_latent_var.shape) * noise_strength
                )

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
                generated_image = generated_image[:, :, :, :image_width]
                generated_image = tf.transpose(generated_image, (0, 2, 3, 1))
                generated_image = (
                    tf.clip_by_value(generated_image, -1.0, 1.0) + 1.0
                ) * 127.5

                loss = perceptual_loss([target_image, generated_image])
            gradients = tape.gradient(loss, [w_latent_var])
            self.optimizer.apply_gradients(zip(gradients, [w_latent_var]))

            loss_tracker.increment_losses({"perceptual_loss": loss})

            if step % self.save_and_log_frequency == 0:
                saved_latents.append(w_latent_var.numpy())
                loss_tracker.print_losses(step)
        infere = Infere()
        infere.genererate_chosen_text(
            [text_of_the_image],
            prefix="projected_image",
            output_path=output_dir,
            sentence=False,
            w_latents=saved_latents[-1],
        )
        with open(os.path.join(output_dir, "latents.txt"), "w") as file:
            for latent in saved_latents:
                file.write(str(latent) + "\n")


if __name__ == "__main__":
    projector = Projector()
    projector.main(TARGET_IMG_PATH, TEXT_ON_IMG, OUTPUT_PATH)
