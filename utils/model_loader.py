import tensorflow as tf
from models.stylegan2.generator import Generator
from models.stylegan2.discriminator import Discriminator

##TODO: add typing
class ModelLoader:
    def initiate_models(self, g_params, d_params):
        discriminator = self._load_discriminator(d_params, ckpt_dir=None)
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
            g_params = {
                "z_dim": 512,
                "w_dim": 512,
                "labels_dim": 0,
                "n_mapping": 8,
                "resolutions": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                "featuremaps": [512, 512, 512, 512, 512, 256, 128, 64, 32],
            }

        test_latent = tf.ones((1, g_params["z_dim"]), dtype=tf.float32)
        test_labels = tf.ones((1, g_params["labels_dim"]), dtype=tf.float32)

        # build generator model
        generator = Generator(g_params)
        _ = generator([test_latent, test_labels])

        if ckpt_dir is not None:
            if is_g_clone:
                ckpt = tf.train.Checkpoint(g_clone=generator)
            else:
                ckpt = tf.train.Checkpoint(generator=generator)
            manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(f"Generator restored from {manager.latest_checkpoint}")
        return generator

    def _load_discriminator(self, d_params=None, ckpt_dir=None):

        if d_params is None:
            d_params = {
                "labels_dim": 0,
                "resolutions": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                "featuremaps": [512, 512, 512, 512, 512, 256, 128, 64, 32],
            }

        res = d_params["resolutions"][-1]
        test_images = tf.ones((1, 3, res, res), dtype=tf.float32)
        test_labels = tf.ones((1, d_params["labels_dim"]), dtype=tf.float32)

        # build discriminator model
        discriminator = Discriminator(d_params)
        _ = discriminator([test_images, test_labels])

        if ckpt_dir is not None:
            ckpt = tf.train.Checkpoint(discriminator=discriminator)
            manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(
                    "Discriminator restored from {}".format(manager.latest_checkpoint)
                )
        return discriminator
