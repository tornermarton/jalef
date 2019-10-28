from abc import ABC, abstractmethod
from typing import Union, Tuple

import cudf
from cudf import DataFrame


class DatasetGenerator(ABC):
    """
    Abstract class to generate datasets for training which can be used easily with this framework.
    """

    def __init__(self,
                 dataset_name: str,
                 datasets_root_path: str,
                 ):
        self._dataset_name: str = dataset_name
        self._datasets_root_path: str = datasets_root_path

    def __count_words_in_sequence(self, sequence):

    @abstractmethod
    def _get_raw_texts(self) -> DataFrame:
        """This method should return the DataFrame containing the list of raw texts."""

        pass

    def generate(self,
                 raw_texts: DataFrame,
                 min_sequence_length: int = 128,
                 max_sequence_length: int = 256,
                 use_top_n_categories: int = None,
                 split_dataset: bool=True
                 ) -> Union[DataFrame, Tuple[DataFrame, DataFrame, DataFrame]]:
        # count words
        return