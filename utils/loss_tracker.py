from time import time
from typing import List
from tensorflow.keras.metrics import Mean
import tensorflow as tf

from config import cfg


class LossTracker(object):
    def __init__(self, loss_names: List[str], print_step=None, log_losses=None):
        self.print_step = print_step
        self.log_losses = log_losses
        self.loss_names = loss_names
        self._initiate_loss_tracking()

    def _initiate_loss_tracking(self):

        self.losses = {
            loss_name: Mean(loss_name, dtype=tf.float32)
            for loss_name in self.loss_names
        }

        self.timer = Mean("timer", dtype=tf.float32)
        self.start_time = time()

    def increment_losses(self, losses: dict):
        for loss_name, loss_value in losses.items():
            if loss_value > 0:
                self.losses[loss_name](loss_value)

        self.timer(time() - self.start_time)
        self.start_time = time()

    def print_losses(self, step):
        start_print = "Step: {}. Avg over the last {:d} steps. {:.2f} s/step. Losses:".format(
            step,
            int(self.timer.count.numpy() / cfg.strategy.num_replicas_in_sync),
            self.timer.result().numpy(),
        )

        loss_print = ", ".join(
            [
                "- {:s}: {:.4f}".format(
                    loss_name, self.losses[loss_name].result().numpy()
                )
                for loss_name in self.loss_names
            ]
        )

        print(start_print + loss_print)

    def reinitialize_tracker(self):
        self._initiate_loss_tracking()
