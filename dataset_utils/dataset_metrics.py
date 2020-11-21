from config import train_cfg as cfg

import os
import cv2
import numpy as np
from tqdm import tqdm


def compute_metrics():
    """
    Computes metric for images depending on their label length (used to design the network)
    :return:
    """

    sizes_info = {
        key: {f"width{key}": [], f"height{key}": [], f"ratios{key}": []}
        for key in range(0, 9)
    }
    sizes_info["9+"] = {"width9+": [], "height9+": [], "ratios9+": []}
    with open(os.path.join(cfg.training_dir, "annotations.txt"), "r") as annotations:
        lines = annotations.readlines()
        for line in tqdm(lines):
            img_name, label = line.split(",", 1)
            img = cv2.imread(os.path.join(cfg.training_dir, img_name))
            h, w, _ = img.shape
            label_len = len(label.strip("\n"))
            key = label_len if label_len <= 8 else "9+"
            sizes_info[key][f"width{key}"].append(w)
            sizes_info[key][f"height{key}"].append(h)
            sizes_info[key][f"ratios{key}"].append(w / h)

    for key in sizes_info.keys():
        print(
            f"Labels of {key} length appear {len(sizes_info[key][f'width{key}'])} in the dataset\n"
            f"The mean ratio of w/h is {np.mean(sizes_info[key][f'ratios{key}'])}\n"
            f"The mean width is {np.mean(sizes_info[key][f'width{key}'])}\n"
            f"The mean height is {np.mean(sizes_info[key][f'height{key}'])}\n"
            f"-------------------------------------------------------------"
        )


if __name__ == "__main__":
    compute_metrics()
