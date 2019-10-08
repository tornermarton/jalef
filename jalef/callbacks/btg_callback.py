from tensorflow.python.keras.callbacks import Callback
import numpy as np
import os

from jalef.plots import plot_confusion_matrix


class BTGCallback(Callback):
    def __init__(self, validation_data, lookup_table, generator_instance, batch_size=32, save_eval_on_epoch_end=True, test_dir_path="test"):
        super().__init__()

        self._test_dir_path = test_dir_path
        self._save_eval_on_epoch_end = save_eval_on_epoch_end
        self._X_validation, self._y_validation = validation_data
        self._generator = generator_instance
        self._batch_size = batch_size
        self._lut = lookup_table

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self._X_validation, verbose=0, batch_size=self._batch_size), axis=1)

        if self._save_eval_on_epoch_end:
            plot_confusion_matrix(y_true=np.array([self._lut.at[e, "Intent"] for e in self._y_validation]),
                                  y_pred=np.array([self._lut.at[e, "Intent"] for e in y_pred]),
                                  title="Epoch: {}".format(epoch),
                                  show_fig=False,
                                  save_fig=True,
                                  filename=os.path.join(self._test_dir_path, "epoch_{}.png".format(epoch)))

            with open(file=os.path.join(self._test_dir_path, "probs.csv"), mode="ab") as file:
                probs = self._generator.get_probs()
                assert type(probs) is np.ndarray

                np.savetxt(file, probs.reshape(1, -1), delimiter=",")

        all_errors = self._y_validation[y_pred != self._y_validation]

        errors_per_class = np.array(
            [len(all_errors[all_errors == k]) / len(all_errors) for k in range(len(np.unique(self._y_validation)))])

        self._generator.update_probs(errors_per_class=errors_per_class)