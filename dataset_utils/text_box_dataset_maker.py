from typing import TextIO
import os
import cv2

from config import cfg
from config.char_tokens import MAIN_CHAR_VECTOR

# Image Datasets
MLT19 = "MLT19"
MLT17 = "MLT17"

def retrieve_latin_text_boxes(data_dir: str, annotation_file: TextIO) -> None:
    files = os.listdir(data_dir)
    assert "gt.txt" in files
    with open(os.path.join(data_dir, "gt.txt")) as gt_file:
        valid_languages = ["Italian", "English", "French", "Latin"]
        img_prefix = os.path.basename(data_dir)
        lines = gt_file.readlines()
        for line in lines:
            img_name, language, label = line.split(",", 2)
            if language in valid_languages and is_label_valid(label):
                new_img_name = f"{img_prefix}_{img_name}"
                # open the file and save it to the right folder instead of simply copying the file
                # to save it in the right format and avoid libpng warning when training
                img = cv2.imread(os.path.join(data_dir, img_name))
                cv2.imwrite(
                    os.path.join(cfg.training_text_boxes_dir, new_img_name), img
                )
                annotation_file.write(f"{new_img_name},{label}")


def main() -> None:

    print("Selecting train images")
    source_datasets = [f"{MLT17}/{MLT17}_1", f"{MLT17}/{MLT17}_2", MLT19]
    source_datasets_path = [os.path.join(cfg.source_datasets, dataset) for dataset in source_datasets]

    with open(
        os.path.join(cfg.training_text_boxes_dir, "annotations.txt"), "w"
    ) as annotation_file:
        [
            retrieve_latin_text_boxes(data_dir, annotation_file)
            for data_dir in source_datasets_path
        ]


def is_label_valid(label: str) -> bool:
    return not any((c not in MAIN_CHAR_VECTOR) for c in label.strip("\n"))


if __name__ == "__main__":

    main()
