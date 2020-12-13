from models.stylegan2.generator import Generator
from models.stylegan2.discriminator import Discriminator
from config import cfg


import tensorflow as tf


class ModelLoader:
    def initiate_models(self):
        discriminator = self._load_discriminator()
        generator = self.load_generator(is_g_clone=False, ckpt_dir=None)
        g_clone = self.load_generator(is_g_clone=True, ckpt_dir=None)

        # set initial g_clone weights same as generator
        g_clone.set_weights(generator.get_weights())
        return discriminator, generator, g_clone

    def load_generator(self, is_g_clone=False, ckpt_dir=None):

        test_latent = tf.ones((1, cfg.z_dim), dtype=tf.float32)
        test_input_text = tf.ones((1,cfg.max_chars), dtype=tf.int32)

        # build generator model
        generator = Generator()
        generator((test_input_text, test_latent))

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

    def _load_discriminator(self):

        res = cfg.discrim_resolutions[0]
        test_images = tf.ones((1, 3, res[0], res[1]), dtype=tf.float32)

        # build discriminator model
        discriminator = Discriminator()
        _ = discriminator(test_images)

        return discriminator

    # TODO check if inference can be faster if better loading
    def load_checkpoint(
        self, ckpt_kwargs, model_description, expect_partial, ckpt_dir, max_to_keep=None
    ):
        ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=max_to_keep)
        if expect_partial:
            ckpt.restore(manager.latest_checkpoint).expect_partial()
        else:
            ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(
                "{} restored from {}".format(
                    model_description, manager.latest_checkpoint
                )
            )
        return manager
