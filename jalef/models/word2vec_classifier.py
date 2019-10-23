from typing import List

import tensorflow as tf

from .engine import SequenceClassifierCore


class Word2VecClassifier(SequenceClassifierCore):
    """A simple classifier architecture using Word2Vec word embeddings."""

    def __init__(self,
                 n_classes: List[int],
                 time_steps: int,
                 fc_layer_sizes: List[int],
                 lstm_layer_sizes: List[int],
                 name: str,
                 logs_root_path: str = "."
                 ):

        super().__init__(n_classes, time_steps, fc_layer_sizes, lstm_layer_sizes, name, logs_root_path)

    def _construct_model(self, print_summary: bool, embedding_matrix=None, **kwargs) -> None:
        """Construct the model architecture.

        Embeddings can be pretrained or also learned during training process.

        :param print_summary: Print model summary after compilation.
        :param embedding_matrix: The embedding matrix to use, if None the weights start as random vectors and are
        learned during training.
        :param kwargs: -
        :return: None
        """

        inputs = tf.keras.layers.Input(shape=(self._time_steps,))

        # If defined, use pretrained word embeddings
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
