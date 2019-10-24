import os

from jalef.plots import plot_confusion_matrix
from .custom_callback import CustomCallback
from .store_validation_results import StoreValidationResults


class SaveConfusionMatrix(CustomCallback):
    def __init__(self, svr_instance: StoreValidationResults, title: str, save_freq: int=1):
        self._svr = svr_instance
        self._title = title

        if not (save_freq > 1 and type(save_freq) is int):
            raise ValueError("Save frequency must be a positive integer!")

        self._save_freq = save_freq

        super().__init__()

    def init_training(self, log_dir_path):
        super().init_training(log_dir_path)

        self._log_dir_path = os.path.join(self._log_dir_path, "confusion_matrices")
        os.makedirs(self._log_dir_path, mode=775)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._save_freq == 0:
            y_true = self._svr.get_true_classes()
            y_pred = self._svr.get_predicted_classes()

            plot_confusion_matrix(y_true=y_true,
                                  y_pred=y_pred,
                                  title=self._title.format(epoch),
                                  show_fig=False,
                                  save_fig=True,
                                  filename=os.path.join(self._log_dir_path, "epoch_{}.png".format(epoch))
                                  )
