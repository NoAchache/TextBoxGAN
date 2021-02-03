import os
from typing import TextIO

import cv2

from config import cfg
from config.char_tokens import MAIN_CHAR_VECTOR

# Image Datasets
MLT19 = "MLT19"
MLT17 = "MLT17"
VALID_LANGUAGES = ["Italian", "English", "French", "Latin"]


def retrieve_latin_text_boxes(data_dir: str, annotation_file: TextIO) -> None:
    """
    Extract the text boxes from the source dataset for which the language is in VALID_LANGUAGES

    Parameters
    ----------
    data_dir: Directory containing the text boxes.
    annotation_file: File where the information on each image kept in the dataset are written.

    """
    files = os.listdir(data_dir)
    assert "gt.txt" in files
    with open(os.path.join(data_dir, "gt.txt")) as gt_file:
        image_prefix = os.path.basename(data_dir)
        lines = gt_file.readlines()
        for line in lines:
            image_name, language, word = line.split(",", 2)
            if language in VALID_LANGUAGES and is_word_valid(word):
                new_image_name = f"{image_prefix}_{image_name}"
                # open the file and save it to the right folder instead of simply copying the file
                # to save it in the right format and avoid libpng warning when training
                image = cv2.imread(os.path.join(data_dir, image_name))
                cv2.imwrite(
                    os.path.join(cfg.training_text_boxes_dir, new_image_name), image
                )
                annotation_file.write(f"{new_image_name},{word}")


def main() -> None:
    """
    Entry point of the file. Creates the text box dataset from MLT 17 and MLT 19

    """

    print("Selecting train images")
    source_datasets = [f"{MLT17}/{MLT17}_1", f"{MLT17}/{MLT17}_2", MLT19]
    source_datasets_path = [
        os.path.join(cfg.source_datasets, dataset) for dataset in source_datasets
    ]

    with open(
        os.path.join(cfg.training_text_boxes_dir, "annotations.txt"), "w"
    ) as annotation_file:
        [
            retrieve_latin_text_boxes(data_dir, annotation_file)
            for data_dir in source_datasets_path
        ]


def is_word_valid(word: str) -> bool:
    return not any((c not in MAIN_CHAR_VECTOR) for c in word.strip("\n"))


if __name__ == "__main__":

    main()
