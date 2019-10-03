from tensorflow.python.keras.callbacks import Callback
import numpy as np


class BTGCallback(Callback):
    def __init__(self, validation_data, generator_instance):
        super().__init__()

        self._X_validation, self._y_validation = validation_data
        self._generator = generator_instance

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self._X_validation, verbose=0), axis=1)

        all_errors = self._y_validation[y_pred != self._y_validation]

        errors_per_class = np.array(
            [len(all_errors[all_errors == k]) / len(all_errors) for k in range(len(np.unique(self._y_validation)))])

        self._generator.update_probs(errors_per_class=errors_per_class)
