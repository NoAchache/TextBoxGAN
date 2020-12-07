import tensorflow as tf
import tensorflow_addons as tfa


from config import cfg

#TODO: changer ca par un model avec call

class AsterInferer:
    def __init__(self, combine_forward_and_backward=False):
        self.combine_forward_and_backward = combine_forward_and_backward
        tfa.register_all()
        self.model = tf.saved_model.load(cfg.aster_weights, tags='serve').signatures['serving_default']

    def run(self, inputs):

        logits = []
        for i in range(len(inputs)):
            prediction = self.model(inputs[i:i + 1])
            if self.combine_forward_and_backward:
                logits.append(self._postprocess_combine(prediction))
            else:
                logits.append(self._postprocess_simple(prediction["forward_logits"]))

        return tf.concat(logits, axis=0)

    def _postprocess_combine(self, prediction):

        #retrieve logits and keep only the first cfg.max_chars time steps
        forward_logits = prediction["forward_logits"][:, :cfg.max_chars]
        backward_logits = prediction["backward_logits"][:, :cfg.max_chars]

        combined_logits = self._combine_logits(forward_logits, backward_logits)

        #retrieve the remaining logits of forward
        remaining_logits = forward_logits[:, combined_logits.shape[1]:, :]

        #compute the required padding so that the output tensor has exactly cfg.max_chars time steps
        padding_len = cfg.max_chars - forward_logits.shape[1]

        # creates a tensor filled with 0s and 1s, to pad the logits with blank indexes. Multiply
        # it by 1000 since the loss uses a softmax
        padding = tf.expand_dims(
                tf.tile(
                        [tf.cast(
                                tf.equal(tf.range(combined_logits.shape[2]), 1),
                                tf.float32)],
                        [padding_len, 1])
                , 0)*1000

        return tf.concat([combined_logits, remaining_logits, padding], axis=1)

    def _combine_logits(self, forward_logits, backward_logits):
        # create masks to filter blank indexes
        forward_mask = ~tf.equal(tf.argmax(forward_logits, axis=2), 1)
        backward_mask = ~tf.equal(tf.argmax(backward_logits, axis=2), 1)

        # filter out blank indexes
        masked_forward = forward_logits[forward_mask]
        masked_backward = backward_logits[backward_mask][::-1]  # reverse it

        # ensure both tensors now have the same shape (requirement of tf.where)
        crop_masked_forward = masked_forward[:masked_backward.shape[0]]
        crop_masked_backward = masked_backward[:masked_forward.shape[0]]

        # get softmax element for each time step
        forward_max = tf.reduce_max(crop_masked_forward, axis=1)
        backward_max = tf.reduce_max(crop_masked_backward, axis=1)

        combined_logits = tf.where(tf.expand_dims(forward_max, 1) > tf.expand_dims(backward_max, 1),
                               crop_masked_forward, crop_masked_backward)

        return tf.expand_dims(combined_logits, 0)


    def _postprocess_simple(self, logits):

        logits = logits[:, :cfg.max_chars]

        padding_len = cfg.max_chars - logits.shape[1]

        if padding_len > 0:
            # creates a tensor filled with 0s and 1s, to pad the logits with blank indexes. Multiply
            # it by 1000 since the loss uses a softmax
            padding = tf.expand_dims(
                        tf.tile(
                                [tf.cast(
                                        tf.equal(tf.range(logits.shape[2]), 1),
                                tf.float32)],
                        [padding_len, 1])
                    , 0)*1000

            logits = tf.concat([logits, padding], axis=1)


        return logits
