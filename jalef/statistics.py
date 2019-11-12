from sklearn import metrics
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import List, Tuple


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


def remove_uncertain_predictions(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 preds: np.ndarray,
                                 threshold: float
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the uncertain predictions (highest prob is under threshold) from the original results. A new dataframe with
    only the certain enough ones is returned.

    :param y_true: List of true values
    :param y_pred: List of predicted values
    :param preds:  List of predictions (probs, network softmax output)
    :param threshold: Threshold
    :return: The true values, the predicted values and the predicted probabilities, where the predictions were certain
    enough. (over threshold)
    """

    ids = [idx for idx, p_probs in enumerate(preds) if max(p_probs) > threshold]

    return y_true[ids], y_pred[ids], preds[ids]


def select_errors(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  preds: np.ndarray,
                  reverse_lut: pd.DataFrame
                  ) -> pd.DataFrame:
    """
    Select the cases where the prediction does not match with the truth. The datadrame returned contains the predicted
    and the true class and the corresponding probs.

    :param y_true: List of true values
    :param y_pred: List of predicted values
    :param preds:  List of predictions (probs, network softmax output)
    :param reverse_lut: The reversed look-up table (class (text)->label (int))
    :return: Dataframe to display
    """

    df = pd.DataFrame(columns=["True", "True %", "Predicted", "Predicted %"])

    for i, t, p, p_probs in zip(range(len(y_true)), y_true, y_pred, preds):
        if t != p:
            df.loc[i] = [t,
                         "{:.2f}".format(p_probs[reverse_lut.loc[t]["Label"]]),
                         p,
                         "{:.2f}".format(p_probs[reverse_lut.loc[p]["Label"]])
                         ]

    return df


def compare_thresholds(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       preds: np.ndarray,
                       thresholds: List[float]
                       ) -> pd.DataFrame:
    """
    Compare the results of using some thresholds with method remove_uncertain_preds(). Some statistics are also returned.

    :param y_true: List of true values
    :param y_pred: List of predicted values
    :param preds:  List of predictions (probs, network softmax output)
    :param thresholds: List of thresholds
    :return: Dataframe to display
    """

    df = pd.DataFrame(
        columns=["Predictions (or.)", "Predictions", "Predictions % (or.)", "Uncertain", "Uncertain % (or.)", "True",
                 "True %", "True % (or.)", "False", "False %", "False % (or.)", "F/T ratio"]
    )

    for t in thresholds:
        y_true_c, y_pred_c, _ = remove_uncertain_predictions(
            y_true=y_true,
            y_pred=y_pred,
            preds=preds,
            threshold=t
        )

        n_removed = len(y_pred) - len(y_pred_c)
        n_err_c = len([i for i, j in zip(y_true_c, y_pred_c) if i != j])
        n_corr_c = len(y_pred_c) - n_err_c

        # df = df.append({'A': i}, ignore_index=True)
        df.loc[t] = ["{:.2f}".format(len(y_true)),
                     "{:.2f}".format(len(y_true_c)),
                     "{:.2f}".format(len(y_true_c) / len(y_true) * 100),
                     "{:.2f}".format(n_removed),
                     "{:.2f}".format(n_removed / len(y_true) * 100),
                     "{:.2f}".format(n_corr_c),
                     "{:.2f}".format(n_corr_c / len(y_true_c) * 100),
                     "{:.2f}".format(n_corr_c / len(y_true) * 100),
                     "{:.2f}".format(n_err_c),
                     "{:.2f}".format(n_err_c / len(y_true_c) * 100),
                     "{:.2f}".format(n_err_c / len(y_true) * 100),
                     "{:.2f}".format(n_err_c / n_corr_c)
                     ]

    def green_color(e):
        return 'color: green'

    def orange_color(e):
        return 'color: orange'

    df = df.style.applymap(green_color,
                           subset=["Predictions (or.)", "Uncertain % (or.)", "True % (or.)", "False % (or.)"]) \
        .applymap(orange_color, subset=["Predictions", "True %", "False %"])

    return df


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
