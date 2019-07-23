#!/usr/bin/env python3

import argparse
import csv
import re
import os
from nltk import tokenize

import pandas as pd

from IPython.display import display

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
    sentences = tokenize.sent_tokenize(text)
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


def create_dataset_from_temp(output_file, min_sequence_length, n, verbosity):
    # read temp dataset
    dataset = pd.read_csv(TEMP_FILE, names=["Text", "Intent"])

    if verbosity >= 1:
        print("The dataset has {} rows.".format(len(dataset)))

    if verbosity >= 2:
        print()
        display(dataset.head(10))

        print()
        print("The following lines have some unexpected errors:")

        # this should produce empty output which means no NaN values
        for e, i in zip(dataset["Text"], dataset["Filename"]):
            if type(e) is not str:
                print(dataset.loc[dataset['Filename'] == i], e, i)

    dataset["Word_count"] = [len(e.split(" ")) for e in dataset["Text"]]

    if verbosity >= 2:
        print()
        display_word_counts(dataset)

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
        print("The new size of the dataframe is {}.".format(len(dataset)))

    if verbosity >= 2:
        print()
        print("The new statistics:")
        display_word_counts(dataset)

    dataset = dataset.loc[:, ["Text", "Intent"]]

    dataset.to_csv(path_or_buf=output_file, header=True, index=False)

    return


def main(args):
    if os.path.exists(TEMP_FILE):
        raise FileExistsError("A file with the name of the temporary file already exists! ({}))".format(TEMP_FILE))

    if os.path.exists(TEMP_FILE):
        raise FileExistsError("A file with the name of the output file already exists!")

    if args.min_seq_len > args.max_seq_len:
        raise ValueError("min_seq_len cannot be bigger than max_seq_len!")

    if args.verbosity >= 1:
        print("Start reading files...")

    create_temp_dataset(data_root_directory=args.data_root_dir, max_sequence_length=args.max_seq_len,
                        verbosity=args.verbosity)

    if args.verbosity >= 1:
        print("Temporary dataset created, start cleaning data...")

    create_dataset_from_temp(output_file=args.output_path, min_sequence_length=args.min_seq_len,
                             n=args.use_top_n_categories, verbosity=args.verbosity)

    if args.verbosity >= 1:
        print("Dataset created, delete temporary dataset...")

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    if args.verbosity >= 1:
        print("OK")

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
                        help="The path of the output file. (The output is in csv format)",
                        default="dataset.csv",
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

    parser.add_argument("-v", "--verbosity",
                        metavar="verbosity",
                        help="The level of verbosity (0 is switched off).",
                        default=1,
                        choices=[0, 1, 2],
                        type=int
                        )

    args = parser.parse_args()

    main(args=args)
