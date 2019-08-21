from typing import Union

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def train_validation_test_split(dataset: Union[list, np.ndarray, pd.DataFrame],
                                validation_size: float,
                                test_size: float):
    """The simple extension of the well-known method defined in sklearn package.

    Splits the data into three subsets (without shuffling).

    :param dataset: The whole dataset as an array.
    :param validation_size: The size of the validation set (0...1).
    :param test_size: The size of the validation set (0...1).
    :return: The three subsets (train, validation, test).
    """

    # compute validation size to use
    v_s = validation_size / (1 - test_size)

    # train-validation-test split
    tmp, test = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=False, stratify=None)
    train, validation = train_test_split(tmp, test_size=v_s, random_state=42, shuffle=False, stratify=None)

    return train, validation, test
