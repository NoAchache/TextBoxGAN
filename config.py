from easydict import EasyDict
import os.path as osp
from datetime import datetime
import tensorflow as tf


train_cfg = EasyDict()
infere_cfg = EasyDict()

shared_cfg = EasyDict()

###Infere###
EXPERIMENT_NAME = ""  # experiment to load from
infere_cfg.ckpt_dir = osp.join("./checkpoints", EXPERIMENT_NAME)
############

###Train###
# TODO: ensure it's not a diff name for everywhere cfg is loaded
train_cfg.experiment_name = f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y|%Hh%M')}"
train_cfg.data_dir = "./data"
train_cfg.source_datasets = osp.join(train_cfg.data_dir, "source_datasets")
train_cfg.training_dir = osp.join(train_cfg.data_dir, "training_data")
train_cfg.ckpt_dir = osp.join("./checkpoints", train_cfg.experiment_name)

train_cfg.shuffle_seed = 4444
###########


shared_cfg.im_size = 256
shared_cfg.z_dim_char = 16  # TODO: rename to latent_size?
shared_cfg.w_dim_char = 16
shared_cfg.batch_size_per_gpu = 16
shared_cfg.strategy = tf.distribute.MirroredStrategy()
shared_cfg.batch_size = (
    shared_cfg.batch_size_per_gpu * shared_cfg.strategy.num_replicas_in_sync
)

# model
shared_cfg.embedding_dim = 32

# text boxes specs
shared_cfg.char_height = 64  # height of a character
shared_cfg.char_width = 32  # width of a character
shared_cfg.min_chars = 1  # min number of chars
shared_cfg.max_chars = 8  # max number of chars

# resources
shared_cfg.num_gpus = 1
shared_cfg.num_workers = 5

train_cfg.update(shared_cfg)
infere_cfg.update(shared_cfg)


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
