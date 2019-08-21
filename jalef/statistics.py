from sklearn import metrics
from pandas import DataFrame
import numpy as np


def display_word_counts(df: DataFrame) -> None:
    """Display the statistics of the word sequences in a pandas DataFrame.

    :param df: The pandas DataFrame.
    :return: None
    """

    min_word_count = min(df["Word_count"])
    min_char_count = min(df["Character_count"])

    max_word_count = max(df["Word_count"])
    max_char_count = max(df["Character_count"])

    print("Max number of words in a sentence: {}".format(max_word_count))
    print("Max number of characters in a sentence: {}".format(max_char_count))

    print("Min number of words in a sentence: {}".format(min_word_count))
    print("Min number of characters in a sentence: {}".format(min_char_count))


def evaluate_result(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the evaluation metrics (accuracy, precision, recall and f1 score) of the test.

    :param y_true: The target values.
    :param y_pred: The predicted values.
    :return: The metrics in a dictionary.
    """

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
