from time import time
from tensorflow.keras.metrics import Mean
import tensorflow as tf


class LossTracker(object):
    def __init__(self, print_step):
        self.print_step = print_step
        self._initiate_loss_tracking()

    def _initiate_loss_tracking(self):
        self.loss_names = [
            "reg_g_loss",
            "g_loss",
            "pl_penalty",
            "ocr_loss",
            "reg_d_loss",
            "d_loss",
            "r1_penalty",
        ]

        self.losses = {loss_name: Mean(loss_name, dtype=tf.float32) for loss_name in self.loss_names}

        self.timer = Mean("timer", dtype=tf.float32)
        self.start_time = time()

    def increment_losses(self, losses: dict):
        for loss_name, loss_value in losses.items():
            self.losses[loss_name](loss_value)

        self.timer(time() - self.start_time)
        self.start_time = time()

    def print_losses(self, step):

        start_print = "Step: {}. Avg over the last {:d} steps. {:.2f} s/step. Losses:".format(
            step, self.timer.count.numpy(), self.timer.result().numpy()
        )

        loss_print = ", ".join(
            [
                "- {:s}: {:.4f}".format(loss_name, self.losses[loss_name].result().numpy())
                for loss_name in self.loss_names
            ]
        )

        print(start_print + loss_print)

    def reinitialize_tracker(self):
        self._initiate_loss_tracking()

    def write_dict(self):
        loss_dict = {
            "total_loss": self.total_losses.avg,
            "segmentation_loss": self.seg_losses.avg,
            "recognition_loss": self.recog_losses.avg,
            "tr_loss": self.tr_losses.avg,
            "tcl_loss": self.tcl_losses.avg,
            "sin_loss": self.sin_losses.avg,
            "cos_loss": self.cos_losses.avg,
            "radii_loss": self.radii_losses.avg,
            "batch_time": self.timer.avg,
        }

        return loss_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
