import tensorflow as tf
import tensorflow_addons as tfa

from config import cfg


class AsterInferer(tf.keras.Model):
    def __init__(self, combine_forward_and_backward=False):
        super(AsterInferer, self).__init__()
        self.combine_forward_and_backward = combine_forward_and_backward
        tfa.register_all(custom_kernels=False)
        self.model = tf.saved_model.load(cfg.aster_weights, tags="serve").signatures[
            "serving_default"
        ]

    def call(self, inputs, training=False, mask=None):
        logits = []
        for i in range(len(inputs)):
            prediction = self.model(inputs[i : i + 1])
            if self.combine_forward_and_backward:
                logits.append(self._postprocess_combine(prediction))
            else:
                logits.append(self._postprocess_simple(prediction["forward_logits"]))

        return tf.concat(logits, axis=0)

    def _postprocess_combine(self, prediction):

        # retrieve logits and keep only the first cfg.max_chars time steps
        forward_logits = prediction["forward_logits"][:, : cfg.max_chars]
        backward_logits = prediction["backward_logits"][:, : cfg.max_chars]

        combined_logits = self._combine_logits(forward_logits, backward_logits)

        # retrieve the remaining logits of forward
        remaining_logits = forward_logits[:, combined_logits.shape[1] :, :]

        # compute the required padding so that the output tensor has exactly cfg.max_chars time steps
        padding_len = cfg.max_chars - forward_logits.shape[1]

        # creates a tensor filled with 0s and 1s, to pad the logits with blank indexes. Multiply
        # it by 1000 since the loss uses a softmax
        padding = (
            tf.expand_dims(
                tf.tile(
                    [
                        tf.cast(
                            tf.equal(tf.range(combined_logits.shape[2]), 1), tf.float32
                        )
                    ],
                    [padding_len, 1],
                ),
                0,
            )
            * 1000
        )

        return tf.concat([combined_logits, remaining_logits, padding], axis=1)

    def _combine_logits(self, forward_logits, backward_logits):
        # create masks to filter blank indexes
        forward_mask = ~tf.equal(tf.argmax(forward_logits, axis=2), 1)
        backward_mask = ~tf.equal(tf.argmax(backward_logits, axis=2), 1)

        # filter out blank indexes
        masked_forward = forward_logits[forward_mask]
        masked_backward = backward_logits[backward_mask][::-1]  # reverse it

        # ensure both tensors now have the same shape (requirement of tf.where)
        crop_masked_forward = masked_forward[: masked_backward.shape[0]]
        crop_masked_backward = masked_backward[: masked_forward.shape[0]]

        # get softmax element for each time step
        forward_max = tf.reduce_max(crop_masked_forward, axis=1)
        backward_max = tf.reduce_max(crop_masked_backward, axis=1)

        combined_logits = tf.where(
            tf.expand_dims(forward_max, 1) > tf.expand_dims(backward_max, 1),
            crop_masked_forward,
            crop_masked_backward,
        )

        return tf.expand_dims(combined_logits, 0)

    def _postprocess_simple(self, logits):

        logits = logits[:, : cfg.max_chars]

        padding_len = cfg.max_chars - tf.shape(logits)[1]

        if padding_len > 0:
            # creates a tensor filled with 0s and 1s, to pad the logits with blank indexes. Multiply
            # it by 1000 since the loss uses a softmax
            padding = (
                tf.expand_dims(
                    tf.tile(
                        [tf.cast(tf.equal(tf.range(logits.shape[2]), 1), tf.float32)],
                        [padding_len, 1],
                    ),
                    0,
                )
                * 1000
            )

            logits = tf.concat([logits, padding], axis=1)

        return logits

    @staticmethod
    def convert_inputs(fake_images, labels, blank_label):
        """
        Convert inputs from the main network (i.e generator/discriminator) input format to the ocr
        input format.
        """
        fake_images = tf.transpose(fake_images, (0, 2, 3, 1))  # B,C,H,W to B,H,W,C

        def resize_image(inputs):
            fake_image, label = inputs

            blank_label_idxs = tf.where(tf.equal(label, blank_label))

            if len(blank_label_idxs) > 0:
                first_blank_idx = blank_label_idxs[0, 0]

                # crop image parts corresponding to blank labels
                w_crop_idx = (first_blank_idx) * cfg.char_width
                fake_image = fake_image[:, :w_crop_idx, :]

            return tf.image.resize(
                fake_image, [cfg.aster_img_dims[0], cfg.aster_img_dims[1]]
            )

        # Aster ocr works better with resized images rather than padded images.
        return tf.map_fn(fn=resize_image, elems=(fake_images, labels), dtype=tf.float32)
