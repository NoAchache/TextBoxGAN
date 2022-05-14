from typing import Tuple

import tensorflow as tf

from config import cfg
from models.stylegan2.discriminator import Discriminator
from models.stylegan2.generator import Generator


class ModelLoader:
    """Loads the different sub models."""

    def initiate_models(self) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        discriminator = self._load_discriminator()
        generator = self.load_generator(is_g_clone=False, ckpt_dir=None)
        g_clone = self.load_generator(is_g_clone=True, ckpt_dir=None)

        # set initial g_clone weights same as generator
        g_clone.set_weights(generator.get_weights())
        return discriminator, generator, g_clone

    def load_generator(
        self, is_g_clone: bool = False, ckpt_dir: str = None
    ) -> tf.keras.Model:

        test_latent = tf.ones((1, cfg.z_dim), dtype=tf.float32)
        test_input_word = tf.ones((1, cfg.max_char_number), dtype=tf.int32)

        # build generator model
        generator = Generator()
        generator((test_input_word, test_latent), batch_size=1)

        if ckpt_dir is not None:
            ckpt_kwargs = (
                {"g_clone": generator} if is_g_clone else {"generator": generator}
            )
            self.load_checkpoint(
                ckpt_kwargs=ckpt_kwargs,
                model_description="Generator",
                expect_partial=True,
                ckpt_dir=ckpt_dir,
            )

        return generator

    def _load_discriminator(self) -> tf.keras.Model:

        res = cfg.discrim_resolutions[0]
        test_images = tf.ones((1, 3, res[0], res[1]), dtype=tf.float32)

        # build discriminator model
        discriminator = Discriminator()
        _ = discriminator(test_images)

        return discriminator

    def load_checkpoint(
        self,
        ckpt_kwargs: dict,
        model_description: str,
        expect_partial: bool,
        ckpt_dir: str,
        max_to_keep=None,
    ) -> tf.train.CheckpointManager:
        ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=max_to_keep)

        resume_checkpoint = (
            f"{cfg.ckpt_dir}/ckpt-{cfg.resume_step}"
            if cfg.resume_step != -1
            else manager.latest_checkpoint
        )

        if expect_partial:
            ckpt.restore(resume_checkpoint).expect_partial()

        else:
            ckpt.restore(resume_checkpoint)
        if manager.latest_checkpoint:
            print("{} restored from {}".format(model_description, resume_checkpoint))
        return manager
