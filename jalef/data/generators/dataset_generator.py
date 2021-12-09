from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import nltk
import csv

import pandas as pd
from pandas import DataFrame

from jalef.preprocessing import train_validation_test_split, Text

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize


class DatasetGenerator(ABC):
    """
    Abstract class to generate datasets for training which can be used easily with this framework.
    """

    TEMP_FILE = ".temp_dataset.csv"

    def __init__(self,
                 source_name: str,
                 datasets_root_path: str,
                 verbosity: int = 0
                 ):
        self._source_name: str = source_name
        self._datasets_root_path: str = datasets_root_path
        self._verbosity = verbosity

    @abstractmethod
    def _get_raw_texts(self,
                       use_top_n_categories: int = None,
                       use_categories: List[str] = None
                       ) -> DataFrame:
        """
        This method should return the DataFrame containing the list of raw texts.
        The two necessary columns are: Text, Label!!!
        The task specific cleaning is also needed to be done at this step. Help: jalef.preprocessing.Text
        """

        pass

    def _clean_raw_text(self,
                        text: str,
                        replace_words: dict
                        ) -> str:

        text = Text(text=text)

        for search, target in replace_words.items():
            text = text.replace_word(search=search, target=target)

        return str(text.clean())

    def _process_raw_text(self,
                          text: str,
                          min_sequence_length: int,
                          max_sequence_length: int,
                          split_up_long_text: bool,
                          replace_words: dict
                          ):

        # Clean the text and change/mask words
        text = self._clean_raw_text(text=text, replace_words=replace_words)

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        lens = [len(s) for s in sentences]

        # Remove too short sequences
        lens = [l for l in lens if l >= min_sequence_length]

        indices = []
        start = end = 0

        if split_up_long_text:
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

                # normal case
                elif (end - start) + l > max_sequence_length:
                    indices.append(slice(start, end, 1))
                    start = end

                end += l

                # last step flush the remaining if any
                if i == len(lens) - 1 and end > start:
                    indices.append(slice(start, end, 1))
        else:
            # Just cut the sentence if too long
            indices.append(slice(0, max_sequence_length, 1))

        return [" ".join(words[ind]) for ind in indices]

    def _process_raw_texts(self,
                           raw_texts: DataFrame,
                           min_sequence_length: int,
                           max_sequence_length: int,
                           split_up_long_texts: bool,
                           replace_words: dict
                           ) -> DataFrame:

        dataset = raw_texts

        for row in raw_texts.itertuples():
            sequences = self._process_raw_text(
                text=row["Text"],
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                split_up_long_text=split_up_long_texts,
                replace_words=replace_words
            )

            records = [row.copy() for _ in range(len(sequences))]

            for i, s in enumerate(sequences):
                records[i]["Text"] = s

            pd.DataFrame.from_records(records).to_csv(DatasetGenerator.TEMP_FILE, mode='a', index=False)

        # remove too short sequences
        dataset["word_count"] = [len(nltk.tokenize.word_tokenize(t)) for t in dataset["Text"]]
        dataset = dataset.loc[dataset["word_count"] >= min_sequence_length]

        # If set, split up too long sentences (the preprocessors shound take care of clipping too long sentences)
        if split_up_long_texts:
            # calculate the number of sequences after the operation to allocate the properly sized dataframe
            n =

            # create lookup tables

    def generate(self,
                 min_sequence_length: int = 128,
                 max_sequence_length: int = 256,
                 use_top_n_categories: int = None,
                 use_categories: List[str] = None,
                 split_up_long_texts: bool = False,
                 split_dataset: bool = True,
                 replace_words: dict = {},
                 ) -> Union[Tuple[DataFrame, DataFrame], Tuple[DataFrame, DataFrame, DataFrame, DataFrame]]:

        # some basic checks
        if min_sequence_length > max_sequence_length:
            raise ValueError("min_seq_len cannot be bigger than max_seq_len!")

        if use_top_n_categories is not None and use_categories is not None:
            raise ValueError("Only one type of category selection can be used!")

        raw_texts = self._get_raw_texts(
            use_categories=use_categories,
            use_top_n_categories=use_top_n_categories
        )

        dataset = self._process_raw_texts(
            raw_texts=raw_texts,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            split_up_long_texts=split_up_long_texts,
            replace_words=replace_words
        )

        lut = {idx: element for idx, element in enumerate(list(set(dataset["Intent"])))}
        reverse_lut = {value: key for key, value in lut.items()}

        dataset["Label"] = [reverse_lut[intent] for intent in dataset["Intent"]]

        # create DataFrame from lut
        lut = pd.DataFrame.from_dict(lut, orient='index')


        if split_dataset:
            return train_validation_test_split(dataset=dataset, validation_size=0.2, test_size=0.1), lut
        else:
            return dataset, lut
