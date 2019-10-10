from tensorflow.python.keras.callbacks import Callback
import os

from jalef.plots import plot_confusion_matrix
from jalef.callbacks import StoreValidationResults


class SaveConfusionMatrix(Callback):
    def __init__(self, svr_instance: StoreValidationResults, title: str, save_directory: str, save_freq: int=1):
        self._svr = svr_instance
        self._title = title

        if not (save_freq > 1 and type(save_freq) is int):
            raise ValueError("Save frequency must be a positive integer!")

        self._save_freq = save_freq

        if os.path.exists(save_directory):
            raise ValueError("Invalid save directory path!")

        self._path = os.path.join(save_directory, "confusion_matrices")

        if not os.path.exists(self._path):
            raise ValueError("Path {} already exists!")

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._save_freq:
            y_true = self._svr.get_true_classes()
            y_pred = self._svr.get_predicted_classes()

            plot_confusion_matrix(y_true=y_true,
                                  y_pred=y_pred,
                                  title=self._title.format(epoch),
                                  show_fig=False,
                                  save_fig=True,
                                  filename=os.path.join(self._path, "epoch_{}.png".format(epoch))
                                  )
