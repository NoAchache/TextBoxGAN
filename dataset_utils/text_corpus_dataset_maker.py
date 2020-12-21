from tqdm import tqdm
import os
import numpy as np

from config import cfg
from config.char_tokens import MAIN_CHAR_VECTOR


def get_words_from_file(file_name: str) -> dict:
    file_path = os.path.join(cfg.source_datasets, file_name)

    words_appearance_per_char = {char: [] for char in MAIN_CHAR_VECTOR}
    with open(file_path, "rb") as file:
        print("Retrieving words from: " + file_name)
        for line in tqdm(file.readlines()):
            try:
                line = line.decode("utf-8")
            except:
                continue

            for word in line.split(" "):
                word = word.strip("\n")
                len_condition = (
                    len(word) <= 8 and len(word) >= 1
                    if file_name == "wikipediaTXT.txt"
                    else len(word) <= 8
                )
                if is_label_valid(word) and len_condition:
                    for letter in word:
                        words_appearance_per_char[letter].append(word)

    return words_appearance_per_char


def select_words(
    english_dict_words_generators: dict,
    wikipedia_words_generators: dict,
    max_words: int,
):
    chars_number_appearance = {char: 0 for char in MAIN_CHAR_VECTOR}

    all_words = []
    english_dict_word = "hello"
    wikipedia_word = "world!"

    special_chars = MAIN_CHAR_VECTOR[MAIN_CHAR_VECTOR.find("Z") + 1 :]

    def add_word(word):
        if word is not None:
            num_special_chars = 0
            len_word = len(word)
            for i, char in enumerate(word):
                if char in "?!,." and i != len_word - 1:
                    word = word.replace("?", "", 1)
                elif char in special_chars:
                    num_special_chars += 1

            if num_special_chars >= 3 or word in all_words:
                return

            all_words.append(word)
            for char in word:
                chars_number_appearance[char] += 1

    if max_words == -1:
        max_words = float("inf")

    while (english_dict_word is not None or wikipedia_word is not None) and len(
        all_words
    ) < max_words:
        add_word(wikipedia_word)
        add_word(english_dict_word)

        least_appeared_char = min(
            chars_number_appearance, key=chars_number_appearance.get
        )

        english_dict_word = next(
            english_dict_words_generators[least_appeared_char], None
        )
        wikipedia_word = next(wikipedia_words_generators[least_appeared_char], None)

    return all_words


def main() -> None:

    english_dict_words_appearance_per_char = get_words_from_file(
        "english_dictionary.txt"
    )
    wikipedia_words_appearance_per_char = get_words_from_file("wikipediaTXT.txt")

    np.random.seed(cfg.shuffle_seed)

    english_dict_words_generators = {}
    wikipedia_words_generators = {}

    for (char, english_dict_words), wikipedia_words in zip(
        english_dict_words_appearance_per_char.items(),
        wikipedia_words_appearance_per_char.values(),
    ):
        english_dict_words = np.array(english_dict_words)
        wikipedia_words = np.array(wikipedia_words)

        np.random.shuffle(english_dict_words)
        np.random.shuffle(wikipedia_words)

        english_dict_words_generators[char] = iter(english_dict_words)
        wikipedia_words_generators[char] = iter(wikipedia_words)

    test_size = cfg.num_test_words
    validation_size = cfg.num_validation_words

    test_words = select_words(
        english_dict_words_generators, wikipedia_words_generators, test_size
    )
    validation_words = select_words(
        english_dict_words_generators, wikipedia_words_generators, validation_size
    )
    train_words = select_words(
        english_dict_words_generators, wikipedia_words_generators, -1
    )

    print(f"The train dataset contains {len(train_words)} words")

    train_file = open(
        os.path.join(cfg.training_text_corpus_dir, "train_corpus.txt"), "w"
    )
    validation_file = open(
        os.path.join(cfg.training_text_corpus_dir, "validation_corpus.txt"), "w"
    )
    test_file = open(os.path.join(cfg.training_text_corpus_dir, "test_corpus.txt"), "w")

    for words, file in zip(
        [test_words, validation_words, train_words],
        [test_file, validation_file, train_file],
    ):
        for word in words:
            file.write(word + "\n")
    train_file.close()
    test_file.close()
    validation_file.close()


def is_label_valid(label: str) -> bool:
    return not any((c not in MAIN_CHAR_VECTOR) for c in label.strip("\n"))


if __name__ == "__main__":

    main()
