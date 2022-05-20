import tensorflow as tf

OLD_AND_NEW_NAMES = {
    "weights": "kernel",
    "fully_connected": "dense",
    "biases": "bias",
    "Predictor/decoder/dense": "Predictor/dense",
    "Backward/Predictor/decoder/sync_attention_wrapper/bahdanau_attention/query_layer": (
        "sync_attention_wrapper_1/BahdanauAttention"
    ),
    "Forward/Predictor/decoder/sync_attention_wrapper/bahdanau_attention/query_layer": (
        "sync_attention_wrapper/BahdanauAttention"
    ),
    "Predictor/decoder/sync_attention_wrapper/lstm_cell": "Predictor/lstm_cell",
    "decoder/sync_attention_wrapper/bahdanau_attention/attention_v": (
        "BahdanauAttention/attention_v"
    ),
    "Predictor/memory_layer": "Predictor/BahdanauAttention",
}

ASTER_ORIGINAL_WEIGHTS = ""  # path to aster weights
ASTER_MODIFIED_WEIGHTS = ""  # local path


def rename_weigths():
    """
    Rename aster layers to switch from tf1 to tf2
    """

    tf1_weights = ASTER_ORIGINAL_WEIGHTS
    tf2_weights = ASTER_MODIFIED_WEIGHTS

    new_vars = []

    with tf.compat.v1.Session() as sess:
        for var_name, _ in tf.train.list_variables(tf1_weights):
            # Load the variable
            var = tf.train.load_variable(tf1_weights, var_name)
            new_name = var_name

            for old, new in OLD_AND_NEW_NAMES.items():
                new_name = new_name.replace(old, new)
            new_vars.append(tf.Variable(var, name=new_name))

        saver = tf.compat.v1.train.Saver(new_vars)
        sess.run(
            [
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ]
        )
        saver.save(sess, tf2_weights)


if __name__ == "__main__":
    rename_weigths()
