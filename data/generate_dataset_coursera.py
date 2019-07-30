#!/usr/bin/env python3

import argparse
import csv
import re
import os
from nltk import tokenize

import pandas as pd
import numpy as np

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

PATTERNS_TO_REMOVE = ["\[(.*?)\]", ">>"]
TEMP_FILE = ".temp_dataset.csv"


class MinValueAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 1:
            parser.error("Minimum value for {0} is 1".format(option_string))

        setattr(namespace, self.dest, values)


def display_word_counts(df):
    min_word_count = min(df["Word_count"])
    min_char_count = min(df["Character_count"])

    max_word_count = max(df["Word_count"])
    max_char_count = max(df["Character_count"])

    print("Max number of words in a sentence: {}".format(max_word_count))
    print("Max number of characters in a sentence: {}".format(max_char_count))

    print("Min number of words in a sentence: {}".format(min_word_count))
    print("Min number of characters in a sentence: {}".format(min_char_count))


def escape_text(raw):
    # make single line
    processed = raw.replace("\n", "")

    # remove patterns
    processed = re.sub("|".join(PATTERNS_TO_REMOVE), "", processed)

    # trim
    processed = processed.strip()

    # reduce multiple spaces
    processed = re.sub(" +", " ", processed)

    return processed


def text_to_sequences(text, max_sequence_length, verbosity):
    while True:
        try:
            sentences = tokenize.sent_tokenize(text)
            break
        except LookupError:
            import nltk
            nltk.download('punkt')

    words = text.split(" ")
    lens = [len(s.split(" ")) for s in sentences]

    if len(lens) == 1 and lens[0] > 2 * max_sequence_length:
        # there is probably something wrong here so we leave this file out
        if verbosity >= 2:
            print("The following text was left out due to some problems, please examine it: {}".format(text))
        return []

    indices = []
    start = 0
    end = 0

    for i, l in enumerate(lens):

        # if a setence is longer than the max_sequence_length we must cut it - spec. case
        if l > max_sequence_length:
            # flush history if any
            if end > start:
                indices.append(slice(start, end, 1))
                start = end

            # add trimmed
            indices.append(slice(start, start + max_sequence_length, 1))
            start = start + l  # end is handled normally

        elif (end - start) + l > max_sequence_length:
            indices.append(slice(start, end, 1))
            start = end

        end += l

        # last step flush the remaining if any
        if i == len(lens) - 1 and end > start:
            indices.append(slice(start, end, 1))

    return [" ".join(words[ind]) for ind in indices]


def process_text(text, max_sequence_length, verbosity):
    escaped = escape_text(raw=text)

    sequences = text_to_sequences(text=escaped, max_sequence_length=max_sequence_length, verbosity=verbosity)

    return sequences


def process_file(root, filename, max_sequence_length, verbosity):
    intent = filename.split("_")[0]

    with open(os.path.join(root, filename)) as in_file:
        text = " ".join(in_file.readlines())

        with open(TEMP_FILE, "a") as out_file:
            writer = csv.writer(out_file)
            writer.writerows(((text, intent) for text in
                              process_text(text=text, max_sequence_length=max_sequence_length,
                                           verbosity=verbosity)))


def create_temp_dataset(data_root_directory, max_sequence_length, verbosity):
    # walk root directory and append processed text (rows) to the output file.
    for root, dirs, files in os.walk(data_root_directory):
        for filename in sorted(files):
            intent = filename.split("_")[0]

            with open(os.path.join(root, filename)) as in_file:
                text = " ".join(in_file.readlines())

                with open(TEMP_FILE, "a") as out_file:
                    writer = csv.writer(out_file)
                    writer.writerows(((text, intent) for text in
                                      process_text(text=text, max_sequence_length=max_sequence_length,
                                                   verbosity=verbosity)))

    return


def create_dataset_from_temp(output_directory, min_sequence_length, n, do_splitting, verbosity):
    # read temp dataset
    dataset = pd.read_csv(filepath_or_buffer=TEMP_FILE, names=["Text", "Intent"])

    if verbosity >= 1:
        print("The dataset has {} rows.".format(len(dataset)))

    if verbosity >= 2:
        print()
        display(dataset.head(10))
        #
        # print()
        # print("The following lines have some unexpected errors:")
        #
        # # this should produce empty output which means no NaN values
        # for e, i in zip(dataset["Text"], dataset["Filename"]):
        #     if type(e) is not str:
        #         print(dataset.loc[dataset['Filename'] == i], e, i)

    dataset["Word_count"] = [len(e.split(" ")) for e in dataset["Text"]]

    original_len = len(dataset)

    # select the top n intents
    dataset = dataset.loc[dataset["Intent"].isin(
        dataset.groupby("Intent").count().sort_values(by=["Text"], ascending=False).index[:n].tolist())]

    if verbosity >= 1:
        print(
            "The result containing only the TOP{} intents has {} rows. It is {:.2f}% of the original dataset.".format(
                n, len(dataset), len(dataset) / original_len * 100))

    # Remove too short sequences
    dataset = dataset.loc[dataset["Word_count"] >= min_sequence_length]

    if verbosity >= 1:
        print("Too short sentences removed. The new size of the dataframe is {}.".format(len(dataset)))

    # generate and save LUT
    lut = {idx: element for idx, element in enumerate(list(set(dataset["Intent"])))}

    reverse_lut = {value: key for key, value in lut.items()}

    dataset["Label"] = [reverse_lut[intent] for intent in dataset["Intent"]]

    pd.DataFrame.from_dict(lut, orient='index').to_csv(path_or_buf=os.path.join(output_directory, "lut.csv"))

    # select only the relevant columns to save
    dataset = dataset.loc[:, ["Text", "Label"]]

    if do_splitting:
        # shuffle database (no need to reset index because it is dropped at saving)
        dataset = shuffle(dataset)

        # train-validation-test split
        tmp, test = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=False, stratify=None)
        train, validation = train_test_split(tmp, test_size=0.2, random_state=42, shuffle=False, stratify=None)

        train.to_csv(path_or_buf=os.path.join(output_directory, "train.csv"), header=True, index=False)
        validation.to_csv(path_or_buf=os.path.join(output_directory, "validation.csv"), header=True, index=False)
        test.to_csv(path_or_buf=os.path.join(output_directory, "test.csv"), header=True, index=False)

    else:
        dataset.to_csv(path_or_buf=os.path.join(output_directory, "dataset.csv"), header=True, index=False)

    return


def generate_dataset(data_root_directory: str,
                     output_directory: str,
                     min_sequence_length: int,
                     max_sequence_length: int,
                     n: int,
                     do_splitting: bool,
                     verbosity: int):

    if os.path.exists(TEMP_FILE):
        raise FileExistsError("A file with the name of the temporary file already exists! ({}))".format(TEMP_FILE))

    if os.path.exists(TEMP_FILE):
        raise FileExistsError("A file with the name of the output file already exists!")

    if not os.path.exists(output_directory):
        raise FileExistsError("Cannot find output_directory!")

    if min_sequence_length > max_sequence_length:
        raise ValueError("min_seq_len cannot be bigger than max_seq_len!")

    if verbosity >= 1:
        print("Start reading files...")

    create_temp_dataset(data_root_directory=data_root_directory, max_sequence_length=max_sequence_length,
                        verbosity=verbosity)

    if args.verbosity >= 1:
        print("Temporary dataset created, start cleaning data...")

    create_dataset_from_temp(output_directory=output_directory, min_sequence_length=min_sequence_length,
                             n=n, do_splitting=do_splitting, verbosity=verbosity)

    if verbosity >= 1:
        print("Dataset created, delete temporary dataset...")

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    if verbosity >= 1:
        print("OK")

    return


def main(args):
    # set the random seeds
    np.random.seed(1234)

    generate_dataset(data_root_directory=args.data_root_dir,
                     output_directory=args.output_path,
                     min_sequence_length=args.min_seq_len,
                     max_sequence_length=args.max_seq_len,
                     n=args.use_top_n_categories,
                     do_splitting=args.do_splitting,
                     verbosity=args.verbosity)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from raw Coursera texts.")

    parser.add_argument("data_root_dir",
                        metavar="data_root_dir",
                        help="The root directory of the consumed data. (Parent folder)",
                        type=str
                        )

    parser.add_argument("output_path",
                        metavar="output_path",
                        help="The path of the dataset root directory. (The outputs are in csv format - "
                             "train/valid/test)",
                        default="dataset",
                        type=str
                        )

    parser.add_argument("--max-seq-len",
                        metavar="max_seq_len",
                        help="Max. sequence length.",
                        default=256,
                        action=MinValueAction,
                        type=int
                        )

    parser.add_argument("--min-seq-len",
                        metavar="min_seq_len",
                        help="Min. sequence length.",
                        default=128,
                        action=MinValueAction,
                        type=int
                        )

    parser.add_argument("--use-top-n-categories",
                        metavar="n",
                        help="Use only the top N categories (intents).",
                        default=None,
                        type=int
                        )

    parser.add_argument("--do-splitting",
                        help="Generate train/valid/test.csv instead of dataset.csv (this shuffles the dataset!)",
                        action="store_true"
                        )

    parser.add_argument("-v", "--verbosity",
                        metavar="verbosity",
                        help="The level of verbosity (0 is switched off).",
                        default=1,
                        choices=[0, 1, 2],
                        type=int
                        )

    args = parser.parse_args()

    main(args=args)
