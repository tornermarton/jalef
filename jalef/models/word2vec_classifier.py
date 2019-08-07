from typing import List

import tensorflow as tf

from .engine import SequenceClassifierCore


class Word2VecClassifier(SequenceClassifierCore):

    def __init__(self,
                 n_classes: List[int],
                 time_steps: int,
                 fc_layer_sizes: List[int],
                 lstm_layer_sizes: List[int],
                 name: str,
                 optimizer: str = "adam",
                 loss: str = "categorical_crossentropy",
                 metrics: list = None,
                 monitor: str = "val_acc",
                 epochs: int = 100,
                 batch_size: int = 256,
                 shuffle: bool = True,
                 patience: int = 10,
                 min_delta: float = 0.005,
                 weights_root: str = ".",
                 tensorboard_root: str = None
                 ):

        super().__init__(n_classes, time_steps, fc_layer_sizes, lstm_layer_sizes, name, optimizer, loss, metrics,
                         monitor, epochs, batch_size, shuffle, patience, min_delta, weights_root, tensorboard_root)

    def _construct_model(self, print_summary: bool, embedding_matrix=None, **kwargs) -> None:
        inputs = tf.keras.layers.Input(shape=(self._time_steps,))

        if embedding_matrix is None:
            x = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          input_length=self._time_steps, trainable=True)(inputs)
        else:
            x = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          input_length=self._time_steps, trainable=False, weights=[embedding_matrix])(inputs)

        # LSTM part
        for i, n in enumerate(self._lstm_layer_sizes):
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n, return_sequences=True))(x)

        # Fix part of the fully-connected part
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation="relu"))(x)
        x = tf.keras.layers.Flatten()(x)

        # Rest of the fully connected part
        for i, n in enumerate(self._fc_layer_sizes):
            x = tf.keras.layers.Dense(n, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)

        outputs = tf.keras.layers.Dense(self._n_classes, activation='softmax')(x)

        self._model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
