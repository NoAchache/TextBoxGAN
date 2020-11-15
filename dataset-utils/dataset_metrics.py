from utils import cfg
import os
import cv2
import numpy as np


def compute_metrics():
    """
    Computes the mean ratios of width over height for images of labels of different length (used to
    design the network)
    :return:
    """
    ratios = {key: [] for key in range(0, 9)}
    ratios["9+"] = []
    with open(os.path.join(cfg.training_dir, "annotations.txt"), "r") as annotations:
        lines = annotations.readlines()
        for line in lines:
            img_name, label = line.split(",", 1)
            img = cv2.imread(os.path.join(cfg.training_dir, img_name))
            h, w, _ = img.shape
            label_len = len(label.strip("\n"))
            ratios[label_len].append(w / h) if label_len <= 8 else ratios["9+"].append(
                w / h
            )

    for key in ratios.keys():
        print(
            f"Labels of {key} length appear {len(ratios[key])} in the dataset and the mean ratio"
            f" of w/h is {np.mean(ratios[key])}"
        )


if __name__ == "__main__":
    compute_metrics()
