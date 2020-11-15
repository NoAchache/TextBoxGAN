from easydict import EasyDict
import os.path as osp


PROJECT_ROOT = "/home/noe/Desktop/Projects/Python/TextBoxGan"

cfg = EasyDict()

cfg.data_dir = osp.join(PROJECT_ROOT, "data")
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
cfg.training_dir = osp.join(cfg.data_dir, "training_data")

cfg.im_size = 256
cfg.latent_size = 512
cfg.batch_size = 16

cfg.shuffle_seed = 4444

# text boxes specs
cfg.char_height = 64  # height of a character
cfg.char_width = 32  # width of a character
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars


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
