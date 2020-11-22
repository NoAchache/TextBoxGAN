from easydict import EasyDict
import os.path as osp
from datetime import datetime
import tensorflow as tf

cfg = EasyDict()
cfg = EasyDict()

cfg = EasyDict()

# Infere
EXPERIMENT_NAME = ""  # experiment to load from
cfg.ckpt_path = osp.join("./checkpoints", EXPERIMENT_NAME)

# Train
# TODO: ensure it's not a diff name for everywhere cfg is loaded
cfg.experiment_name = f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y|%Hh%M')}"
cfg.data_dir = "./data"
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
cfg.training_dir = osp.join(cfg.data_dir, "training_data")
cfg.ckpt_dir = osp.join("./checkpoints", cfg.experiment_name)
cfg.shuffle_seed = 4444

# Text boxes specs
cfg.im_width = 256
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars
cfg.char_height = 64  # height of a character
cfg.char_width = cfg.im_width / cfg.max_chars  # width of a character

# Model
cfg.embedding_dim = 32
cfg.expand_char_w_res = [cfg.char_width / 2, cfg.char_width]
cfg.expand_char_feat_maps = [512, 512]
cfg.expand_word_h_res = [1, 2, 4, 8, 32, 64]
cfg.expand_word_feat_maps = [512, 256, 256, 128, 128, 64]
cfg.z_dim = 512
cfg.style_dim = 512
cfg.n_mapping = 5

cfg.encoded_char_width = (
    cfg.char_width / 4
)  # width of chars after reshaping their encoding

# Resources
cfg.num_gpus = 1
cfg.num_workers = 5

cfg.batch_size_per_gpu = 16
cfg.strategy = tf.distribute.MirroredStrategy()
cfg.batch_size = cfg.batch_size_per_gpu * cfg.strategy.num_replicas_in_sync


# cha = 24
#
# n_layers = int(log2(im_size) - 1)
#
# mixed_prob = 0.9


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")
