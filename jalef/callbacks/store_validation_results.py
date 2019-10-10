from tensorflow.python.keras.callbacks import Callback
import numpy as np
from enum import Enum, auto

from jalef.statistics import evaluate_result


class StoreValidationResults(Callback):
    class Task(Enum):
        REGRESSION = auto()
        CLASSIFICATION = auto()

    def __init__(self, x, y, lookup_fn=None, task: Task = Task.CLASSIFICATION, batch_size=32, show_metrics=True):
        self.__show_metrics = show_metrics

        self._x = x
        self._y_true = y

        self._lookup_fn = lookup_fn

        self._y_pred = None
        self._task = task
        self._metrics = None
        self._batch_size = batch_size

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self._y_pred = self.model.predict(self._X_validation, verbose=0, batch_size=self._batch_size)

        if self.__show_metrics:
            self._metrics = evaluate_result(y_pred=self.get_predicted_classes(), y_true=self.get_true_classes())

            line = "Validation metrics: "
            keys = list(self._metrics.keys())
            for key in keys[:-1]:
                line += "{}={}, ".format(key, self._metrics[key])
            line += "{}={}".format(keys[-1], self._metrics[keys[-1]])

            print(line)

    def get_predicted_classes(self, pretty=False):
        assert self._task == StoreValidationResults.Task.CLASSIFICATION, "Only possible if task is classification!"

        if pretty and self._lookup_fn is not None:
            return np.array([self._lookup_fn(e) for e in np.argmax(a=self._y_pred, axis=1)])
        else:
            return np.argmax(a=self._y_pred, axis=1)

    def get_true_classes(self, pretty=False):
        assert self._task == StoreValidationResults.Task.CLASSIFICATION, "Only possible if task is classification!"

        if pretty and self._lookup_fn is not None:
            return np.array([self._lookup_fn(e) for e in np.argmax(a=self._y_true, axis=1)])
        else:
            return np.argmax(a=self._y_true, axis=1)

    def get_predictions(self):
        return self._y_pred

    def get_trues(self):
        return self._y_true
