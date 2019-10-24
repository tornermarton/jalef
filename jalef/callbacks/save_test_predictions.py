import numpy as np
import os

from .custom_callback import CustomCallback


class SaveTestPredictions(CustomCallback):
    def __init__(self, X: np.ndarray):
        super().__init__()

        self._X = X

    def on_train_end(self, logs=None):
        # load best weights if saved
        best_weights_path = os.path.join(self._log_dir_path, "weights.npy")
        if os.path.exists(best_weights_path):
            self.model.load_weights(filepath=best_weights_path)

        preds = self.model.predict(x=self._X)

        np.save(os.path.join(self._log_dir_path, "predictions.npy"), preds)

        super().on_train_end(logs)
