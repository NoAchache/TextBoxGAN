SHELL := /bin/bash

create-data-dir:
	mkdir data && \
	mkdir data/source_datasets && \
	mkdir data/training_data && \
	mkdir data/training_data/text_boxes && \
	mkdir data/training_data/text_corpus

download-text-datasets:
	cd data/source_datasets && \
	wget http://redac.univ-tlse2.fr/corpus/wikipedia/wikipediaFR-TXT.txt.7z && \
	p7zip -d wikipediaFR-TXT.txt.7z && \
	wget https://codeload.github.com/dwyl/english-words/zip/master && \
	unzip master && \
	mv english-words-master/words.txt english_dictionary.txt

download-image-datasets:
	cd data/source_datasets && \
	mkdir MLT17 && \
	mkdir MLT17/MLT17_1 && \
	mkdir MLT17/MLT17_2 && \
	mkdir MLT19 && \
	cd MLT17 && \
	wget http://datasets.cvc.uab.es/rrc/ch8_training_word_images_gt_part_1.zip && \
	wget http://datasets.cvc.uab.es/rrc/ch8_training_word_images_gt_part_2.zip && \
	wget http://datasets.cvc.uab.es/rrc/ch8_training_word_images_gt_part_3.zip && \
	wget https://rrc.cvc.uab.es/downloads/ch8_validation_word_images_gt.zip --no-check-certificate && \
	unzip ch8_training_word_images_gt_part_1.zip -d MLT17_1 && \
	unzip ch8_training_word_images_gt_part_2.zip -d MLT17_1 && \
	unzip ch8_training_word_images_gt_part_3.zip -d MLT17_1 && \
	unzip ch8_validation_word_images_gt.zip -d MLT17_2 && \
	cd ../MLT19 && \
	wget http://datasets.cvc.uab.es/rrc/words_part_1.zip && \
	wget http://datasets.cvc.uab.es/rrc/words_part_2.zip && \
	wget http://datasets.cvc.uab.es/rrc/words_part_3.zip && \
	unzip words_part_1.zip && \
	unzip words_part_2.zip && \
	unzip words_part_3.zip

make-datasets:
	poetry run python -m dataset_utils.text_corpus_dataset_maker && \
	poetry run python -m dataset_utils.text_box_dataset_maker && \
	poetry run python -m dataset_utils.filter_out_bad_images

download-and-make-datasets: create-data-dir download-text-datasets download-image-datasets make-datasets

black:
	poetry run black .

isort:
	poetry run isort .

#e.g. `make tensorboard xps="xp1 xp2"` to show the logs of xp1 and xp2 in tensorboard
tensorboard:
	for xp in $(xps) ; do \
		concat_xps+=$$xp:experiments/$$xp,; \
	done;\
	concat_xps="$${concat_xps%?}";\
	tensorboard --logdir_spec "$$concat_xps" --samples_per_plugin "text=400,images=400"
