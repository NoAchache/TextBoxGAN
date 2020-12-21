from easydict import EasyDict
import os.path as osp
import os
from datetime import datetime
import tensorflow as tf

from config.char_tokens import CharTokenizer

cfg = EasyDict()

cfg.working_dir = os.path.dirname(os.path.dirname(__file__))
cfg.experiment_dir = osp.join(cfg.working_dir, "experiments")

EXPERIMENT_NAME = "Long_run"  # experiment to load from

cfg.experiment_name = (
    f"TextBoxGan_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"
    if EXPERIMENT_NAME == None
    else EXPERIMENT_NAME
)
cfg.ckpt_dir = osp.join(cfg.experiment_dir, cfg.experiment_name, "checkpoints")
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
cfg.min_chars = 1  # min number of chars
cfg.max_chars = 8  # max number of chars
cfg.image_width = cfg.char_width * cfg.max_chars

# Model
cfg.embedding_out_dim = 32
cfg.word_encoder_dense_dim = 256

cfg.generator_resolutions = [
    (2, 8),  # (h, w)
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
]

generator_initial_feat_maps = int(
    cfg.word_encoder_dense_dim
    * cfg.max_chars
    / (cfg.generator_resolutions[0][0] * cfg.generator_resolutions[0][1])
)

cfg.generator_feat_maps = [generator_initial_feat_maps, 512, 256, 256, 128, 128]
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
    == (cfg.char_height, cfg.image_width)
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
cfg.ocr_loss = 0.0001

# Logging, Summary, Save
cfg.summary_steps_frequency = {"print_steps": [50, 500], "log_losses": [False, True]}
cfg.image_summary_step_frequency = 199

cfg.num_images_per_log = 3
cfg.validation_step_frequency = 10000
cfg.save_step_frequency = 5000
cfg.num_ckpts_to_keep = 5

# Resources
cfg.num_workers = tf.data.experimental.AUTOTUNE
cfg.strategy = tf.distribute.MirroredStrategy()
cfg.batch_size_per_gpu = 16
cfg.batch_size = cfg.batch_size_per_gpu * cfg.strategy.num_replicas_in_sync

# OCR
cfg.aster_weights = osp.join(cfg.working_dir, "aster_weights")
cfg.aster_img_dims = (64, 256)
cfg.ocr_loss = "softmax_crossentropy"
assert cfg.ocr_loss in ["softmax_crossentropy", "mse"]

# Others
cfg.shuffle_seed = 4444
cfg.max_steps = 10e7
cfg.char_tokenizer = CharTokenizer()


def print_config(config):
    print("==========Options============")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print("=============End=============")
