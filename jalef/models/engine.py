import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Union
import numpy as np
import json

from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from tensorflow.python.keras.callbacks import History

from tensorflow.python.keras import Model

from jalef.callbacks import CustomCallback


class Core(ABC):
    """The core for all models used in this package.

    This is a basic wrapper for tf.keras models to generalize the training scripts and make it easier to switch
    between various models.

    Usage:
    In this example MyModel is derived from the Core class.

    ...
    model = MyModel("mymodel", "/app/logs/weights/")
    model.compile(loss="categorical_crossentropy")
    model.train(X_train, y_train, X_valid, y_valid, X_test, y_test)
    ...

    For more information please read the projects README.
    """

    def __init__(self,
                 name: str = "jalef_model",
                 logs_root_path: str = None
                 ):

        self._name: str = name

        self._callbacks: List[Callback] = []
        self._custom_callbacks: List[Callback] = []

        self._logs_root_path = logs_root_path
        self._weights_path = None

        self._model: Model = None

    def add_custom_callback(self, callback: CustomCallback) -> None:
        """Add custom callback to use when training model."""

        self._custom_callbacks.append(callback)

    def __init_training(self, monitor, patience, min_delta) -> str:
        training_name: str = self._name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir_path: str = os.path.join(self._logs_root_path, training_name)
        os.makedirs(log_dir_path)

        # define log directories and create them
        weights_path = os.path.join(log_dir_path, "weights.hdf5")
        self._weights_path = weights_path
        model_configs_path = os.path.join(log_dir_path, "model_configs.json")
        training_log_path = os.path.join(log_dir_path, "training.log")
        tensorboard_path = os.path.join(os.path.join(self._logs_root_path, "tensorboard"), training_name)

        # save model architecture
        with open(model_configs_path, "w") as file:
            json.dump(obj=self._model.to_json(), fp=file)

        # create basic callbacks which are used always
        model_checkpoint: Callback = ModelCheckpoint(filepath=weights_path,
                                                     monitor=monitor,
                                                     verbose=1,
                                                     save_best_only=True)

        early_stopping: Callback = EarlyStopping(patience=patience,
                                                 min_delta=min_delta,
                                                 verbose=1, monitor=monitor)

        tensorboard: Callback = TensorBoard(log_dir=tensorboard_path)

        csv_logger: Callback = CSVLogger(filename=training_log_path, append=True)

        self._callbacks = [csv_logger, tensorboard, model_checkpoint, early_stopping]

        # initialize all added custom callbacks
        for c in self._custom_callbacks:
            c.init_training(log_dir_path=log_dir_path)

        return log_dir_path

    @abstractmethod
    def _construct_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model architecture.

        This method must be implemented in all models, it is used to define the model structure.
        The method must set self._model to determine which architecture is to be compiled and trained.

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        pass

    def compile(self,
                optimizer: str = "adam",
                loss: str = "mse",
                metrics: List[str] = None,
                print_summary: bool = True,
                **kwargs) -> None:

        """Construct and compile model.

        ModelCheckpoint and EarlyStopping are added automatically, tensorboard is
        optional, if you don't want to use it leave tensorboard_root on None.

        Since this is a tf.keras model wrapper class, the parameters can be given the same way as there.

        :param print_summary: Print model summary after compilation.
        :param kwargs: Further parameters passed to the _constuct_model method which is class specific.
        :return: None
        """

        self._construct_model(print_summary=print_summary, **kwargs)
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_valid: np.ndarray,
              y_valid: np.ndarray,
              monitor: str = "val_acc",
              patience: int = 10,
              min_delta: float = 0.005,
              epochs: int = 10,
              batch_size: int = 32,
              shuffle: bool = True,
              verbose: int = 0) -> History:

        """Start training process, run tests and save results.

        Since this is a tf.keras model wrapper class, the parameters can be given the same way as there.

        :param verbose: Set the verbosity.
        :return: Network history.
        """

        log_dir_path = self.__init_training(monitor=monitor, patience=patience, min_delta=min_delta)

        print("The logs are saved under the following path: {}".format(log_dir_path))

        return self._model.fit(X_train, y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose,
                               validation_data=(X_valid, y_valid),
                               shuffle=shuffle,
                               # concatenate callbacks list
                               callbacks=(self._custom_callbacks + self._callbacks)
                               )

    def train_generator(self,
                        generator,
                        X_valid: np.ndarray,
                        y_valid: np.ndarray,
                        monitor: str = "val_acc",
                        patience: int = 10,
                        min_delta: float = 0.005,
                        epochs: int = 10,
                        verbose: int = 0) -> History:

        """Start training process, run tests and save results.

        Since this is a tf.keras model wrapper class, the parameters can be given the same way as there.

        :param verbose: Set the verbosity.
        :return: Network history.
        """

        log_dir_path = self.__init_training(monitor=monitor, patience=patience, min_delta=min_delta)

        print("The logs are saved under the following path: {}".format(log_dir_path))

        return self._model.fit_generator(generator=generator,
                                         epochs=epochs,
                                         verbose=verbose,
                                         validation_data=(X_valid, y_valid),
                                         callbacks=self._callbacks
                                         )

    def load_weights_from_file(self, path: str) -> None:
        """Load the network weights from a path.

        :param path: The full path of the weights file.
        :return: None
        """
        self._model.load_weights(filepath=path)

    def load_best_model(self) -> None:
        """Load back the best model. Only works after training!

        :return: None
        """

        assert self._weights_path is not None, "Cannot load back best weights without training."
        self.load_weights_from_file(path=self._weights_path)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Any:
        """Do the evaluation process.

        :param X_test: The test inputs.
        :param y_test: The test outputs (target).
        :return: The output of the keras evaluate method.
        """

        return self._model.evaluate(x=X_test, y=y_test)

    def predict(self, X_test: Union[np.ndarray, List[np.ndarray]], save_predictions: bool = False,
                path: str = "") -> np.ndarray:
        """Return the predictions for the given set. If desired the predictions are saved automatically.

        :param X_test: The test inputs.
        :param save_predictions: Save predictions to the given file.
        :param path: The full path to the output file.
        :return: The predictions.
        """

        preds = self._model.predict(x=X_test)

        if save_predictions:
            np.save(path, preds)

        return preds


class AttentionModelCore(Core, ABC):
    """Base class for all models using attention."""

    def __init__(self,
                 use_attention: bool = True,
                 use_shared_attention_vector: bool = True,
                 name: str = "jalef_attention_model",
                 logs_root_path: str = "",
                 ):
        super().__init__(name, logs_root_path)

        self._use_attention: bool = use_attention
        self._use_shared_attention_vector: bool = use_shared_attention_vector


class Seq2SeqCore(AttentionModelCore, ABC):
    """Base class for all models using a Seq2Seq architecture."""

    def __init__(self,
                 time_steps: int = 50,
                 source_vocab_size: int = 20000,
                 embedding_dim: int = 300,
                 target_vocab_size: int = 20000,
                 trainable_embeddings: bool = False,
                 bidirectional_encoder: bool = True,
                 source_embedding_matrix=None,
                 target_embedding_matrix=None,
                 use_attention: bool = True,
                 use_shared_attention_vector: bool = True,
                 name: str = "jalef_seq2seq_model",
                 logs_root_path: str = ""
                 ):
        super().__init__(use_attention, use_shared_attention_vector, name, logs_root_path)

        self._time_steps: int = time_steps
        self._source_vocab_size: int = source_vocab_size
        self._embedding_dim: int = embedding_dim
        self._trainable_embeddings: bool = trainable_embeddings
        self._target_vocab_size: bool = target_vocab_size
        self._bidirectional_encoder: bool = bidirectional_encoder

        self._source_embedding_matrix = source_embedding_matrix
        self._target_embedding_matrix = target_embedding_matrix

        # Assigned at compilation
        self._encoder_inf_model: Model = None
        self._decoder_inf_model: Model = None

    @abstractmethod
    def _construct_train_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model used at training.

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        pass

    @abstractmethod
    def _construct_inference_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model used at inference (e.g. use in production).

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        pass

    def _construct_model(self, print_summary: bool, **kwargs) -> None:
        """Here this method only calls the two method which are to define the model used at training and the other
        one used at inference."""

        self._construct_train_model(print_summary=print_summary, **kwargs)

        self._construct_inference_model(print_summary=print_summary, **kwargs)


class SequenceClassifierCore(Core, ABC):
    """Base class for all classifier models."""

    def __init__(self,
                 n_classes: List[int],
                 time_steps: int,
                 fc_layer_sizes: List[int],
                 lstm_layer_sizes: List[int],
                 name: str = "jalef_classifier_model",
                 logs_root_path: str = ""
                 ):

        super().__init__(name, logs_root_path)

        self._n_classes: int = n_classes
        self._time_steps = time_steps

        if type(fc_layer_sizes) is not list:
            fc_layer_sizes = [fc_layer_sizes]

        self._fc_layer_sizes: List[int] = fc_layer_sizes

        if type(lstm_layer_sizes) is not list:
            lstm_layer_sizes = [lstm_layer_sizes]

        self._lstm_layer_sizes: List[int] = lstm_layer_sizes
