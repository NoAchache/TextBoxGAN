from easydict import EasyDict
import os.path as osp
import os
from datetime import datetime
import tensorflow as tf

from config.char_tokens import CharTokenizer

cfg = EasyDict()

WORKING_DIR = os.getcwd()
cfg.experiment_dir = osp.join(WORKING_DIR, "experiments")

# Infere files
EXPERIMENT_NAME = ""  # experiment to load from
cfg.ckpt_path = osp.join(cfg.experiment_dir, EXPERIMENT_NAME, "checkpoints")

# Train files
cfg.experiment_name = f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y|%Hh%M')}"
cfg.ckpt_dir = osp.join(cfg.experiment_dir, cfg.experiment_name, "checkpoints")
cfg.log_dir = osp.join(cfg.experiment_dir, cfg.experiment_name, "logs")
cfg.data_dir = osp.join(WORKING_DIR, "data")
cfg.source_datasets = osp.join(cfg.data_dir, "source_datasets")
cfg.training_dir = osp.join(cfg.data_dir, "training_data")

# Text boxes specs
cfg.im_width = 256
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars
cfg.char_height = 64  # height of a character (i.e. height of the image)
cfg.char_width = int(cfg.im_width / cfg.max_chars)  # width of a character

# Model
cfg.embedding_out_dim = 32
cfg.encoding_dense_dim = 256

cfg.generator_resolutions = [
    (2, 8),  # (h, w)
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
]

init_feat_maps = int(
    cfg.encoding_dense_dim
    * cfg.max_chars
    / (cfg.generator_resolutions[0][0] * cfg.generator_resolutions[0][1])
)


cfg.generator_feat_maps = [init_feat_maps, 512, 256, 256, 128, 128]
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


assert (
    cfg.generator_resolutions[-1]
    == cfg.discrim_resolutions[0]
    == (cfg.char_height, cfg.im_width)
)


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
cfg.summary_steps = {"print_steps": [50, 500], "log_losses": [False, True]}
cfg.image_summary_step = 200
cfg.num_images_per_log = 3
cfg.save_step = 500

# Resources
cfg.num_gpus = 1
cfg.num_workers = 5
cfg.strategy = tf.distribute.MirroredStrategy()
cfg.batch_size = 2
cfg.global_batch_size = cfg.batch_size * cfg.strategy.num_replicas_in_sync

# Aster (OCR)
cfg.aster_weights = osp.join(WORKING_DIR, "aster_weights")
cfg.aster_img_dims = (64, 256)

# Others
cfg.shuffle_seed = 4444
cfg.max_epochs = 100000
cfg.char_tokenizer = CharTokenizer()


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")
