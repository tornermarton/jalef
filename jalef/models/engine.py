from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np

import tensorflow as tf
from enum import Enum, auto

from tensorflow.python.keras.callbacks import History


class Core(ABC):

    def __init__(self,
                 optimizer: str = "adam",
                 loss: str = "mse",
                 metrics: list = None,
                 monitor: str = "val_acc",
                 epochs: int = 10,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005
                 ):
        self._optimizer = optimizer
        self._loss = loss

        if metrics is None:
            metrics = ['loss']
        self._metrics = metrics

        self._monitor = monitor

        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._patience = patience
        self._min_delta = min_delta

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self._weights_file, monitor=self._monitor,
                                                              verbose=1, save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self._patience, min_delta=self._min_delta,
                                                          monitor=self._monitor)

        self._callbacks = [model_checkpoint, early_stopping]
        self._weights_file = None
        self._model = None

    @abstractmethod
    def _construct_model(self, print_summary: bool) -> None:
        pass

    def compile(self, print_summary: bool = True, weights_file: str = None) -> None:
        self._construct_model(print_summary=print_summary)

        if weights_file is not None:
            if weights_file != self._weights_file:
                self._weights_file = weights_file
            else:
                print("File exists, it will be overwritten!")
        else:
            print("No weights path specified, tmp file will be used (if exists previous model will be overwritten).")

        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

        if print_summary:
            self._model.summary()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
              verbose: int = 0) -> History:

        history = self._model.fit(X_train, y_train,
                                  batch_size=self._batch_size,
                                  epochs=self._epochs,
                                  verbose=verbose,
                                  validation_data=(X_valid, y_valid),
                                  shuffle=self._shuffle,
                                  callbacks=self._callbacks
                                  )

        return history

    def load_weights(self, filepath: str) -> None:
        self._model.load_weights(filepath=filepath)

    def load_best_model(self) -> None:
        self.load_weights(filepath=self._weights_file)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Any:
        self.load_best_model()

        return self._model.evaluate(x=X_test, y=y_test)

    def add_callback(self, callback: tf.python.keras.callbacks.Callback) -> None:
        self._callbacks.append(callback)

    @abstractmethod
    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        pass


class AttentionModelCore(Core, ABC):

    def __init__(self,
                 use_attention: bool = True,
                 use_shared_attention_vector: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self._use_attention = use_attention
        self._use_shared_attention_vector = use_shared_attention_vector


class Seq2SeqCore(AttentionModelCore, ABC):

    def __init__(self,
                 time_steps: int = 50,
                 source_vocab_size: int = 20000,
                 embedding_dim: int = 300,
                 target_vocab_size: int = 20000,
                 trainable_embeddings: bool = False,
                 use_attention: bool = True,
                 use_shared_attention_vector: bool = True,
                 bidirectional_encoder: bool = True,
                 source_embedding_matrix=None,
                 target_embedding_matrix=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._time_steps = time_steps
        self._source_vocab_size = source_vocab_size
        self._embedding_dim = embedding_dim
        self._trainable_embeddings = trainable_embeddings
        self._target_vocab_size = target_vocab_size
        self._use_attention = use_attention
        self._use_shared_attention_vector = use_shared_attention_vector
        self._bidirectional_encoder = bidirectional_encoder

        self._source_embedding_matrix = source_embedding_matrix
        self._target_embedding_matrix = target_embedding_matrix

        # Assigned at compilation
        self._encoder_inf_model = None
        self._decoder_inf_model = None

    @abstractmethod
    def _construct_train_model(self, print_summary: bool) -> None:
        pass

    @abstractmethod
    def _construct_inference_model(self, print_summary: bool) -> None:
        pass

    def _construct_model(self, print_summary: bool) -> None:
        self._construct_train_model(print_summary=print_summary)

        self._construct_inference_model(print_summary=print_summary)


class CustomModelCore(Core, ABC):
    class Task(Enum):
        REGRESSION = auto()
        CLASSIFICATION = auto()

    def __init__(self,
                 task: Task,
                 n_classes: int,
                 **kwargs):

        super().__init__(**kwargs)

        if task not in CustomModelCore.Task:
            raise ValueError("Task must be one of the supported (eg. .Task.REGRESSION)!")

        self._task = task

        if n_classes < 2:
            raise ValueError("Value of classes must be at least 2!")

        self._n_classes = n_classes


class CustomLSTMModelCore(Core, ABC):

    def __init__(self,
                 lstm_units_size: List[int],
                 **kwargs):
        super().__init__(**kwargs)

        if type(lstm_units_size) is not list:
            lstm_units_size = [lstm_units_size]

        self._lstm_units_size = lstm_units_size
