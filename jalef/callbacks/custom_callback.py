from tensorflow.python.keras.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self):
        super().__init__()

        self._log_dir_path = None

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        assert self._log_dir_path is not None, "Please call init_training() to specify log directory"

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

        self._log_dir_path = None

    def init_training(self, log_dir_path):
        self._log_dir_path = log_dir_path
