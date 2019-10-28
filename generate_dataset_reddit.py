#!/usr/bin/env python3

import argparse
import os
import nltk
from nltk import tokenize

import pandas as pd
import numpy as np

from IPython.display import display
from sklearn.utils import shuffle

from jalef.preprocessing import train_validation_test_split
from jalef.database import DatabaseConnection, Cursor
from jalef.preprocessing import Text

nltk.download('punkt')


class MinValueAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 1:
            parser.error("Minimum value for {0} is 1".format(option_string))

        setattr(namespace, self.dest, values)


def text_to_sequences(text, max_sequence_length, verbosity):
    sentences = tokenize.sent_tokenize(text)
    words = tokenize.word_tokenize(text)
    lens = [len(tokenize.word_tokenize(s)) for s in sentences]

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
    sequences = text_to_sequences(text=text, max_sequence_length=max_sequence_length, verbosity=verbosity)

    return sequences


def get_top_n_categories(n: int) -> np.ndarray:
    # Note that not LIMIT x is used to allow the usage of None to get all of them

    with DatabaseConnection(
            host="172.17.0.3",
            user="tm",
            password="FlDa3846",
            database="reddit"
    ) as connection:
        with Cursor(connection) as cursor:
            cursor.execute(
                "SELECT count(*), symbol, min(name) FROM submissions GROUP BY symbol ORDER BY count(symbol) DESC")

            results = cursor.fetchall()

            symbols = np.empty([len(results)], dtype=object)

            for i, r in enumerate(results):
                symbols[i] = r[1]

            return symbols[:n]


def create_dataset(output_directory, min_sequence_length, n, do_splitting, verbosity):
    symbols = get_top_n_categories(n=n).tolist()

    with DatabaseConnection(
            host="172.17.0.3",
            user="tm",
            password="FlDa3846",
            database="reddit"
    ) as connection:

        with Cursor(connection) as cursor:
            try:
                cursor.execute(
                    "SELECT * FROM submissions WHERE symbol IN ({})".format(", ".join(["%s"] * len(symbols))), symbols)

                df = pd.DataFrame(np.array(cursor.fetchall()), columns=cursor.column_names)

                aggregation_functions = {"id": 'min', 'subreddit': 'min', 'symbol': lambda tdf: tdf.unique().tolist(),
                                         "title": "min", "timestamp": "min", "name": "min", "content": "min"}

                df = df.groupby(['reddit_id']).aggregate(aggregation_functions)
                df.reset_index(level=["reddit_id"], inplace=True)

                df = df[df["content"] != "[deleted]"]

                # use the texts which have only one intent
                df = df[df["symbol"].apply(lambda x: len(x) == 1)]
                df["symbol"] = df["symbol"].apply(lambda x: x[0])

                # clean text and remove symbol
                df["content"] = [str(Text(t).replace_word(s, "symbol").replace_word(n, "Company").clean())
                                 for t, s, n in df[["content", "symbol", "name"]].values]

                dataset = df[["content", "symbol", "timestamp"]].rename(
                    columns={"content": "Text", "symbol": "Intent", "timestamp": "Timestamp"})

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

                dataset["Word_count"] = [len(tokenize.word_tokenize(e)) for e in dataset["Text"]]

                original_len = len(dataset)

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

                pd.DataFrame.from_dict(lut, orient='index').to_csv(
                    path_or_buf=os.path.join(output_directory, "lut.csv"))

                # select only the relevant columns to save
                dataset = dataset.loc[:, ["Text", "Label", "Timestamp"]]

                if do_splitting:
                    # shuffle database (no need to reset index because it is dropped at saving)
                    dataset = shuffle(dataset)

                    # train-validation-test split
                    train, validation, test = train_validation_test_split(dataset=dataset,
                                                                          validation_size=0.2,
                                                                          test_size=0.1
                                                                          )

                    train.to_csv(path_or_buf=os.path.join(output_directory, "train.csv"), header=True, index=False)
                    validation.to_csv(path_or_buf=os.path.join(output_directory, "validation.csv"), header=True,
                                      index=False)
                    test.to_csv(path_or_buf=os.path.join(output_directory, "test.csv"), header=True, index=False)

                else:
                    dataset.to_csv(path_or_buf=os.path.join(output_directory, "dataset.csv"), header=True, index=False)

            except Exception as e:
                print(e)
                return None


def generate_dataset(output_directory: str,
                     min_sequence_length: int,
                     max_sequence_length: int,
                     n: int,
                     do_splitting: bool,
                     verbosity: int):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if min_sequence_length > max_sequence_length:
        raise ValueError("min_seq_len cannot be bigger than max_seq_len!")

    if verbosity >= 1:
        print("Starting process...")

    create_dataset(output_directory=output_directory, min_sequence_length=min_sequence_length,
                   n=n, do_splitting=do_splitting, verbosity=verbosity)

    if verbosity >= 1:
        print("OK")

    return


def main(args):
    # set the random seeds
    np.random.seed(1234)

    generate_dataset(output_directory=args.output_path,
                     min_sequence_length=args.min_seq_len,
                     max_sequence_length=args.max_seq_len,
                     n=args.use_top_n_categories,
                     do_splitting=args.do_splitting,
                     verbosity=args.verbosity)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from Reddit submissions in sql database.")

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
                        default=128,
                        action=MinValueAction,
                        type=int
                        )

    parser.add_argument("--min-seq-len",
                        metavar="min_seq_len",
                        help="Min. sequence length.",
                        default=32,
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
