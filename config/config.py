import os
import os.path as osp
from datetime import datetime

import tensorflow as tf
from easydict import EasyDict

from config.char_tokens import CharTokenizer

""" Configs for training and inference. """

cfg = EasyDict()

cfg.working_dir = os.path.dirname(os.path.dirname(__file__))
cfg.experiment_dir = osp.join(cfg.working_dir, "experiments")

EXPERIMENT_NAME = None  # experiment to load from

cfg.experiment_name = (
    f"TextBoxGAN_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"
    if EXPERIMENT_NAME == None
    else EXPERIMENT_NAME
)
cfg.ckpt_dir = osp.join(cfg.experiment_dir, cfg.experiment_name, "checkpoints")
cfg.resume_step = (
    -1
)  # set it to -1 to select last checkpoint. E.g. resume_step = 225000 to load the file ckpt-225000
cfg.log_dir = osp.join(cfg.experiment_dir, cfg.experiment_name, "logs")

cfg.data_dir = osp.join(cfg.working_dir, "data")
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
training_dir = osp.join(cfg.data_dir, "training_data")
cfg.training_text_boxes_dir = osp.join(training_dir, "text_boxes")
cfg.training_text_corpus_dir = osp.join(training_dir, "text_corpus")
cfg.num_validation_words = 5000
cfg.num_test_words = 5000


# Text boxes specs
cfg.char_height = 64  # height of a character (i.e. height of the image)
cfg.char_width = 32  # width of a character
cfg.max_char_number = 8  # max number of chars

# Model
cfg.embedding_out_dim = 32  # word encoder embedding
cfg.word_encoder_dense_dim = 256

cfg.generator_resolutions = [  # (h, w)
    (2, 8),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),  # This must be equal to (cfg.char_width, cfg.image_width)
]

cfg.generator_feat_maps = [
    None,
    512,
    256,
    256,
    128,
    128,
]  # The first value is computed at the end of the config file
cfg.discrim_resolutions = [  # (h, w)
    (64, 256),  # This must be equal to (cfg.char_width, cfg.image_width)
    (32, 128),
    (16, 64),
    (8, 32),
    (8, 16),
    (4, 8),
    (4, 4),
]
cfg.discrim_feat_maps = [64, 128, 128, 256, 256, 512, 512]

cfg.z_dim = 512
cfg.style_dim = 512
cfg.n_mapping = 5

# Optimizers
cfg.g_opt = {
    "learning_rate": 0.002,
    "beta1": 0.0,
    "beta2": 0.99,
    "epsilon": 1e-08,
    "reg_interval": 8,
}
cfg.d_opt = {
    "learning_rate": 0.002,
    "beta1": 0.0,
    "beta2": 0.99,
    "epsilon": 1e-08,
    "reg_interval": 16,
}

# Logging, Summary, Save
cfg.summary_steps_frequency = {"print_steps": [50, 500], "log_losses": [False, True]}
cfg.image_summary_step_frequency = 500

cfg.num_images_per_log = 3
cfg.validation_step_frequency = 10000
cfg.save_step_frequency = 10000
cfg.num_ckpts_to_keep = 5
cfg.batch_size_per_gpu = 4


# OCR
cfg.aster_weights = osp.join(cfg.working_dir, "aster_weights")
cfg.ocr_loss_weight = 0.0001
cfg.ocr_loss_type = "softmax_crossentropy"
assert cfg.ocr_loss_type in ["softmax_crossentropy", "mse"]

# Others
cfg.shuffle_seed = 4444
cfg.buffer_size = (
    -1
)  # buffer size for the training dataset. Use -1 to select the entire dataset
cfg.max_steps = 130000


#### FIXED CONFIGS, DO NOT CHANGE THEM #####
cfg.image_width = cfg.char_width * cfg.max_char_number

cfg.aster_image_dims = (64, 256)

cfg.char_tokenizer = CharTokenizer()

# cfg.generator_resolutions[0][0] and cfg.generator_resolutions[0][1] corresponds to the height and width of the word
# encoder's output. generator_initial_feat_maps hence corresponds to the number of feature maps of this output.
generator_initial_feat_maps = int(
    cfg.word_encoder_dense_dim
    * cfg.max_char_number
    / (cfg.generator_resolutions[0][0] * cfg.generator_resolutions[0][1])
)

cfg.generator_feat_maps[0] = generator_initial_feat_maps


cfg.num_workers = tf.data.experimental.AUTOTUNE
cfg.strategy = tf.distribute.MirroredStrategy()
cfg.batch_size = cfg.batch_size_per_gpu * cfg.strategy.num_replicas_in_sync
cfg.cpu_only = len(tf.config.list_physical_devices("GPU")) == 0


assert (
    cfg.generator_resolutions[-1]
    == cfg.discrim_resolutions[0]
    == (cfg.char_height, cfg.image_width)
)


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")
