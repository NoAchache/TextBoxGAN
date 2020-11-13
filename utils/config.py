from easydict import EasyDict
import os.path as osp


PROJECT_ROOT = ''

cfg = EasyDict()

cfg.im_size = 256
cfg.latent_size = 512
cfg.batch_size = 16

cha = 24

n_layers = int(log2(im_size) - 1)

mixed_prob = 0.9

