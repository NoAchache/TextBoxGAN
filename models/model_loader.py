from models.stylegan2.generator import Generator
from models.stylegan2.discriminator import Discriminator

import tensorflow as tf


class ModelLoader:
    def initiate_models(self, g_params, d_params):
        discriminator = self._load_discriminator(d_params)
        generator = self._load_generator(
            g_params=g_params, is_g_clone=False, ckpt_dir=None,
        )
        g_clone = self._load_generator(
            g_params=g_params, is_g_clone=True, ckpt_dir=None,
        )

        # set initial g_clone weights same as generator
        g_clone.set_weights(generator.get_weights())
        return discriminator, generator, g_clone

    def _load_generator(self, g_params=None, is_g_clone=False, ckpt_dir=None):

        if g_params is None:
            g_params = {  # TODO: get those from cfg
                "z_dim": 512,
                "w_dim": 512,
                "n_mapping": 8,
                "resolutions": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                "featuremaps": [512, 512, 512, 512, 512, 256, 128, 64, 32],
            }

        test_latent = tf.ones((1, g_params["z_dim"]), dtype=tf.float32)

        # build generator model
        generator = Generator(g_params)
        _ = generator(test_latent)

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

    def _load_discriminator(self, d_params=None):

        if d_params is None:
            d_params = {
                "resolutions": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                "featuremaps": [512, 512, 512, 512, 512, 256, 128, 64, 32],
            }

        res = d_params["resolutions"][-1]
        test_images = tf.ones((1, 3, res, res), dtype=tf.float32)

        # build discriminator model
        discriminator = Discriminator(d_params)
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
