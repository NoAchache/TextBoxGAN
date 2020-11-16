from easydict import EasyDict
import os.path as osp
from datetime import datetime
import tensorflow as tf


PROJECT_ROOT = "/home/noe/Desktop/Projects/Python/TextBoxGan"

cfg = EasyDict()

cfg.experiment_name = f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y|%Hh%M')}"

cfg.data_dir = osp.join(PROJECT_ROOT, "data")
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
cfg.training_dir = osp.join(cfg.data_dir, "training_data")

cfg.im_size = 256
cfg.z_dim = 512  # TODO: rename to latent_size
cfg.batch_size_per_gpu = 16

cfg.shuffle_seed = 4444

cfg.strategy = tf.distribute.MirroredStrategy()

# text boxes specs
cfg.char_height = 64  # height of a character
cfg.char_width = 32  # width of a character
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars

# resources
cfg.num_gpus = 1
cfg.num_workers = 5


# cha = 24
#
# n_layers = int(log2(im_size) - 1)
#
# mixed_prob = 0.9


def initiate_config(cfg):
    cfg.batch_size = cfg.batch_size_per_gpu * cfg.strategy.num_replicas_in_sync


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")
