from tensorflow.python.keras.callbacks import Callback
import numpy as np
import os

from jalef.callbacks import StoreValidationResults
from jalef.generators import BalancedTrainGenerator


class BTGCallback(Callback):
    def __init__(self, svr_instance: StoreValidationResults,
                 generator_instance: BalancedTrainGenerator,
                 save_probs_on_epoch_end: bool = True,
                 test_dir_path: str = "test"
                 ):
        super().__init__()

        self._svr = svr_instance
        self._test_dir_path = test_dir_path
        self._save_probs_on_epoch_end = save_probs_on_epoch_end
        self._generator = generator_instance

    def on_epoch_end(self, epoch, logs=None):
        y_pred_c = self._svr.get_predicted_classes()
        y_true_c = self._svr.get_true_classes()

        if self._save_probs_on_epoch_end:
            with open(file=os.path.join(self._test_dir_path, "probs.csv"), mode="ab") as file:
                probs = self._generator.get_probs()
                assert type(probs) is np.ndarray

                np.savetxt(file, probs.reshape(1, -1), delimiter=",")

        all_errors = y_true_c[y_pred_c != y_true_c]

        errors_per_class = np.array(
            [len(all_errors[all_errors == k]) / len(all_errors) for k in range(len(np.unique(y_true_c)))])

        self._generator.update_probs(errors_per_class=errors_per_class)
