import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Union
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
# from enum import Enum, auto

from tensorflow.python.keras.callbacks import History


class Core(ABC):

    def __init__(self,
                 name: str = "jalef_model",
                 optimizer: str = "adam",
                 loss: str = "mse",
                 metrics: List[str] = None,
                 monitor: str = "val_acc",
                 epochs: int = 10,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005,
                 weights_root: str = "",
                 tensorboard_root: str = None
                 ):

        self._name: str = name
        self._optimizer: str = optimizer
        self._loss: str = loss

        if metrics is None:
            metrics = ['loss']
        self._metrics: List[str] = metrics

        self._monitor: str = monitor

        self._epochs: int = epochs
        self._batch_size: int = batch_size
        self._shuffle: bool = shuffle
        self._patience: int = patience
        self._min_delta: float = min_delta
        self._weights_path: str = os.path.join(weights_root, name + "_weights.hdf5")

        model_checkpoint: Callback = ModelCheckpoint(filepath=self._weights_path,
                                                     monitor=self._monitor,
                                                     verbose=1,
                                                     save_best_only=True)

        early_stopping: Callback = EarlyStopping(patience=self._patience,
                                                 min_delta=self._min_delta,
                                                 verbose=1, monitor=self._monitor)

        self._callbacks: List[Callback] = [model_checkpoint, early_stopping]

        if tensorboard_root is not None:
            tensorboard = TensorBoard(
                log_dir=os.path.join(tensorboard_root, name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")))

            self._callbacks.append(tensorboard)

        self._model: tf.keras.Model = None

    @abstractmethod
    def _construct_model(self, print_summary: bool, **kwargs) -> None:
        pass

    def compile(self, print_summary: bool = True, **kwargs) -> None:
        self._construct_model(print_summary=print_summary, **kwargs)

        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

    def add_callback(self, callback: tf.python.keras.callbacks.Callback) -> None:
        self._callbacks.append(callback)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_valid: np.ndarray,
              y_valid: np.ndarray,
              X_test: np.ndarray,
              y_test: np.ndarray,
              load_best_model_on_end: bool = True,
              evaluate_on_end: bool = True,
              save_predictions_on_end = True,
              verbose: int = 0) -> History:

        history: History = self._model.fit(X_train, y_train,
                                           batch_size=self._batch_size,
                                           epochs=self._epochs,
                                           verbose=verbose,
                                           validation_data=(X_valid, y_valid),
                                           shuffle=self._shuffle,
                                           callbacks=self._callbacks
                                           )

        if load_best_model_on_end:
            self.load_best_model()

        if evaluate_on_end:
            self.evaluate(X_test=X_test, y_test=y_test)

        if save_predictions_on_end:
            self.predict(X_test=X_test, save_predictions=True)

        return history

    def load_weights_from_file(self, path: str) -> None:
        self._model.load_weights(filepath=path)

    def load_best_model(self) -> None:
        self.load_weights_from_file(path=self._weights_path)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Any:
        return self._model.evaluate(x=X_test, y=y_test)

    def predict(self, X_test: Union[np.ndarray, List[np.ndarray]], save_predictions: bool = False,
                path: str = "") -> np.ndarray:
        preds = self._model.predict(x=X_test)

        if save_predictions:
            np.save(path, preds)

        return preds


class AttentionModelCore(Core, ABC):

    def __init__(self,
                 use_attention: bool = True,
                 use_shared_attention_vector: bool = True,
                 name: str = "jalef_model",
                 optimizer: str = "adam",
                 loss: str = "mse",
                 metrics: list = None,
                 monitor: str = "val_acc",
                 epochs: int = 10,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005,
                 weights_root: str = "",
                 tensorboard_root: str = None
                 ):
        super().__init__(name, optimizer, loss, metrics, monitor, epochs, batch_size, shuffle, patience, min_delta, weights_root, tensorboard_root)

        self._use_attention: bool = use_attention
        self._use_shared_attention_vector: bool = use_shared_attention_vector


class Seq2SeqCore(AttentionModelCore, ABC):

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
                 name: str = "jalef_model",
                 optimizer: str = "adam",
                 loss: str = "mse",
                 metrics: list = None,
                 monitor: str = "val_acc",
                 epochs: int = 10,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005,
                 weights_root: str = "",
                 tensorboard_root: str = None
                 ):
        super().__init__(use_attention, use_shared_attention_vector, name, optimizer, loss, metrics, monitor, epochs,
                         batch_size, shuffle, patience, min_delta, weights_root, tensorboard_root)

        self._time_steps: int = time_steps
        self._source_vocab_size: int = source_vocab_size
        self._embedding_dim: int = embedding_dim
        self._trainable_embeddings: bool = trainable_embeddings
        self._target_vocab_size: bool = target_vocab_size
        self._bidirectional_encoder: bool = bidirectional_encoder

        self._source_embedding_matrix = source_embedding_matrix
        self._target_embedding_matrix = target_embedding_matrix

        # Assigned at compilation
        self._encoder_inf_model: tf.keras.Model = None
        self._decoder_inf_model: tf.keras.Model = None

    @abstractmethod
    def _construct_train_model(self, print_summary: bool, **kwargs) -> None:
        pass

    @abstractmethod
    def _construct_inference_model(self, print_summary: bool, **kwargs) -> None:
        pass

    def _construct_model(self, print_summary: bool, **kwargs) -> None:
        self._construct_train_model(print_summary=print_summary, **kwargs)

        self._construct_inference_model(print_summary=print_summary, **kwargs)


class SequenceClassifierCore(Core, ABC):

    def __init__(self,
                 n_classes: List[int],
                 time_steps: int,
                 fc_layer_sizes: List[int],
                 lstm_layer_sizes: List[int],
                 name: str = "jalef_model",
                 optimizer: str = "adam",
                 loss: str = "mse",
                 metrics: list = None,
                 monitor: str = "val_acc",
                 epochs: int = 10,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005,
                 weights_root: str = "",
                 tensorboard_root: str = None
                 ):

        super().__init__(name, optimizer, loss, metrics, monitor, epochs, batch_size, shuffle, patience, min_delta, weights_root, tensorboard_root)

        self._n_classes: int = n_classes
        self._time_steps = time_steps

        if type(fc_layer_sizes) is not list:
            fc_layer_sizes = [fc_layer_sizes]

        self._fc_layer_sizes: List[int] = fc_layer_sizes

        if type(lstm_layer_sizes) is not list:
            lstm_layer_sizes = [lstm_layer_sizes]

        self._lstm_layer_sizes: List[int] = lstm_layer_sizes

#
# class CustomModelCore(Core, ABC):
#     class Task(Enum):
#         REGRESSION = auto()
#         CLASSIFICATION = auto()
#
#     def __init__(self,
#                  task: Task,
#                  n_classes: int,
#                  **kwargs):
#
#         super().__init__(**kwargs)
#
#         if task not in CustomModelCore.Task:
#             raise ValueError("Task must be one of the supported (eg. .Task.REGRESSION)!")
#
#         self._task = task
#
#         if n_classes < 2:
#             raise ValueError("Value of classes must be at least 2!")
#
#         self._n_classes = n_classes
