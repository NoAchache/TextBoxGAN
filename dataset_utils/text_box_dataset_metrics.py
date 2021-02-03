import os

import cv2
import numpy as np
from tqdm import tqdm

from config import cfg
from config.char_tokens import MAIN_CHAR_VECTOR


def compute_metrics():
    """
    Computes two metrics on the text box images:
    - number of text boxes having each number of letters
    - number of occurrences of each letters

    """

    sizes_info = {
        key: {f"width{key}": [], f"height{key}": [], f"ratios{key}": []}
        for key in range(0, cfg.max_char_number + 1)
    }

    chars_info = {char: 0 for char in MAIN_CHAR_VECTOR}

    with open(
        os.path.join(cfg.training_text_boxes_dir, "annotations_filtered.txt"), "r"
    ) as annotations:
        lines = annotations.readlines()
        for line in tqdm(lines):
            image_name, word = line.split(",", 1)
            image = cv2.imread(os.path.join(cfg.training_text_boxes_dir, image_name))
            h, w, _ = image.shape
            word = word.strip("\n")
            word_len = len(word)
            sizes_info[word_len][f"width{word_len}"].append(w)
            sizes_info[word_len][f"height{word_len}"].append(h)
            sizes_info[word_len][f"ratios{word_len}"].append(w / h)

            for char in word:
                chars_info[char] += 1

    for key in sizes_info.keys():
        print(
            f"Labels of {key} length appear {len(sizes_info[key][f'width{key}'])} in the dataset\n"
            f"The mean ratio of w/h is {np.mean(sizes_info[key][f'ratios{key}'])}\n"
            f"The mean width is {np.mean(sizes_info[key][f'width{key}'])}\n"
            f"The mean height is {np.mean(sizes_info[key][f'height{key}'])}\n"
            f"-------------------------------------------------------------"
        )

    for char, num_appearance in chars_info.items():
        print(f"{char} appears {num_appearance} times")


if __name__ == "__main__":
    compute_metrics()
