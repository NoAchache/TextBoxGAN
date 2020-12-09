from utils import cfg
from utils.char_tokens import MAIN_CHAR_VECTOR
from typing import List, TextIO
import os
import zipfile
import shutil
import cv2

# TODO: add wget for zip files

# Datasets
MLT19 = "MLT19"
MLT17 = "MLT17"


class TextBoxDatasetMaker:
    def _extract_dataset(self, dataset_name: str) -> List[str]:
        """
        :param dataset_name: name of the dataset
        :return: path to the directories where the data were extracted
        """
        dataset_dir = os.path.join(cfg.source_datasets, dataset_name)
        zip_files = os.listdir(dataset_dir)
        if dataset_name == MLT19:
            [
                self._unzip_file(os.path.join(dataset_dir, zip_f), dataset_dir)
                for zip_f in zip_files
            ]
            return [dataset_dir]
        elif dataset_name == MLT17:
            # put training and validation data in different folders to avoid naming conflicts,
            # although they will both be used for training
            training_dir1 = os.path.join(dataset_dir, MLT17 + "_1")
            training_dir2 = os.path.join(dataset_dir, MLT17 + "_2")
            for zip_f in zip_files:
                self._unzip_file(
                    os.path.join(dataset_dir, zip_f), training_dir1
                ) if "training" in zip_f else self._unzip_file(
                    os.path.join(dataset_dir, zip_f), training_dir2
                )
            return [training_dir1, training_dir2]

    def _unzip_file(self, zip_path: str, output_dir: str) -> None:
        if os.path.splitext(zip_path)[1] != ".zip":
            return
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_dir)

    def _retrieve_latin_text_boxes(
        self, data_dir: str, annotation_file: TextIO
    ) -> None:
        files = os.listdir(data_dir)
        assert "gt.txt" in files
        with open(os.path.join(data_dir, "gt.txt")) as gt_file:
            valid_languages = ["Italian", "English", "French", "Latin"]
            img_prefix = os.path.basename(data_dir)
            lines = gt_file.readlines()
            for line in lines:
                img_name, language, label = line.split(",", 2)
                if language in valid_languages and not any(
                    (c not in MAIN_CHAR_VECTOR) for c in label.strip("\n")
                ):
                    new_img_name = f"{img_prefix}_{img_name}"
                    # open the file and save it to the right folder instead of simply copying the file
                    # to save it in the right format and avoid libpng warning when training
                    img = cv2.imread(os.path.join(data_dir, img_name))
                    cv2.imwrite(os.path.join(cfg.training_dir, new_img_name), img)
                    annotation_file.write(f"{new_img_name},{label}")

    def main(self) -> None:
        if len(os.listdir(cfg.training_dir)) > 0:
            # dataset_utils already created
            return
        output_dirs_17 = self._extract_dataset(MLT17)
        output_dirs_19 = self._extract_dataset(MLT19)
        annotation_file = open(os.path.join(cfg.training_dir, "annotations.txt"), "w")

        [
            self._retrieve_latin_text_boxes(data_dir, annotation_file)
            for data_dir in output_dirs_17 + output_dirs_19
        ]

        annotation_file.close()


if __name__ == "__main__":
    t = TextBoxDatasetMaker()
    t.main()
