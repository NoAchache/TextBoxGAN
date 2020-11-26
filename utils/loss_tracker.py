from time import time


class LossTracker(object):
    def __init__(self):
        self.log_template = "{:s}, {:s}, {:s}".format(
            "step {}: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}",
            "d_gan_loss: {:.3f}, g_gan_loss: {:.3f}",
            "r1_penalty: {:.3f}, pl_penalty: {:.3f}",
        )

        self.initiate_loss_tracking()
        self.counter = 0

    def initiate_loss_tracking(self):
        self.gen_total_loss = AverageMeter()
        self.seg_loss = AverageMeter()
        self.recog_losses = AverageMeter()
        self.tr_losses = AverageMeter()
        self.tcl_losses = AverageMeter()
        self.sin_losses = AverageMeter()
        self.cos_losses = AverageMeter()
        self.radii_losses = AverageMeter()

        self.timer = AverageMeter()
        self.start_time = time()

    def _update_loss(self, attribute, loss):
        if loss.item() > 0:
            attribute.update(loss.item())

    def increment_losses(self, all_losses):
        assert len(all_losses) == 8

        self._update_loss(self.total_losses, all_losses[0])
        self._update_loss(self.seg_losses, all_losses[1])
        self._update_loss(self.recog_losses, all_losses[2])
        self._update_loss(self.tr_losses, all_losses[3])
        self._update_loss(self.tcl_losses, all_losses[4])
        self._update_loss(self.sin_losses, all_losses[5])
        self._update_loss(self.cos_losses, all_losses[6])
        self._update_loss(self.radii_losses, all_losses[7])

        self.timer.update(time() - self.start_time)
        self.start_time = time()

        self.counter += 1

    def print_losses(self, epoch, ite, len_loader):
        print(
            "epoch: {} ({:d} / {:d}). Avg over the last {:d} steps. {:.2f} s/step. Losses: - Total: {:.4f} - Segmentation: {:.4f} - Recognition: {:.4f} - tr: {:.4f} - tcl: {:.4f} "
            "- sin: {:.4f} - cos: {:.4f} - radii: {:.4f}".format(
                epoch,
                ite,
                len_loader,
                self.counter,
                self.timer.avg,
                self.total_losses.avg,
                self.seg_losses.avg,
                self.recog_losses.avg,
                self.tr_losses.avg,
                self.tcl_losses.avg,
                self.sin_losses.avg,
                self.cos_losses.avg,
                self.radii_losses.avg,
            )
        )

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
