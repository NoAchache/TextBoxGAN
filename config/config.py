from easydict import EasyDict
import os.path as osp
import os
from datetime import datetime
import tensorflow as tf

from config.char_tokens import CharTokenizer

cfg = EasyDict()

WORKING_DIR = os.getcwd()

# Infere files
EXPERIMENT_NAME = ""  # experiment to load from
cfg.ckpt_path = osp.join(WORKING_DIR, "checkpoints", EXPERIMENT_NAME)

# Train files
# TODO: ensure it's not a diff name for everywhere cfg is loaded
# TODO: reorganize files experiments qui contient logs et weights
cfg.experiment_name = f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y|%Hh%M')}"
cfg.data_dir = osp.join(WORKING_DIR, "data")
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
cfg.training_dir = osp.join(cfg.data_dir, "training_data")
cfg.ckpt_dir = osp.join(WORKING_DIR, "checkpoints", cfg.experiment_name)
cfg.log_dir = osp.join(WORKING_DIR, "logs", cfg.experiment_name)

# Text boxes specs
cfg.im_width = 256
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars
cfg.char_height = 64  # height of a character (i.e. height of the image)
cfg.char_width = int(cfg.im_width / cfg.max_chars)  # width of a character

# Model
cfg.embedding_out_dim = 32
cfg.expand_char_w_res = [int(cfg.char_width / 2), cfg.char_width]
cfg.expand_char_feat_maps = [512, 512]
cfg.expand_word_h_res = [1, 2, 4, 8, 32, 64]
cfg.expand_word_feat_maps = [512, 256, 256, 128, 128, 64]
cfg.discrim_resolutions = [
    (64, 256),  # (h, w)
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

cfg.encoded_char_width = (
    cfg.char_width / 4
)  # width of chars after reshaping their encoding

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
cfg.summary_steps = {"print_steps": [1, 500], "log_losses": [False, True]}
cfg.image_summary_step = 1
cfg.num_images_per_log = 3
cfg.save_step = 1
#
# cfg.summary_steps = {"print_steps": [10, 500], "log_losses": [False, True]}
# cfg.image_summary_step = 100
# cfg.num_images_per_log = 3
# cfg.save_step = 500

# Resources
cfg.num_gpus = 1
cfg.num_workers = 5
cfg.strategy = tf.distribute.MirroredStrategy()
cfg.batch_size_per_gpu = 1
cfg.batch_size = cfg.batch_size_per_gpu * cfg.strategy.num_replicas_in_sync

# Aster (OCR)
cfg.aster_weights = osp.join(WORKING_DIR, "aster_weights")
cfg.aster_img_dims = (64, 256)

# Others
cfg.shuffle_seed = 4444
cfg.max_epochs = 100000
cfg.char_tokenizer = (
    CharTokenizer()
)  # TODO check not instantiated every call with print


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")

