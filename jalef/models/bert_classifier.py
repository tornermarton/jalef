from typing import List

import tensorflow as tf

from jalef.layers import Bert
from .engine import SequenceClassifierCore


class BertClassifier(SequenceClassifierCore):
    """A simple classifier architecture using BERT as word embeddings."""

    def __init__(self,
                 pretrained_model_path,
                 output_size,
                 n_layers_to_finetune,
                 n_classes: List[int],
                 time_steps: int,
                 fc_layer_sizes: List[int],
                 lstm_layer_sizes: List[int],
                 name: str,
                 weights_root: str = "."
                 ):
        super().__init__(n_classes, time_steps, fc_layer_sizes, lstm_layer_sizes, name,  weights_root)

        self._pretrained_model_path = pretrained_model_path
        self._output_size = output_size
        self._n_layers_to_finetune = n_layers_to_finetune

    def _construct_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model architecture using BERT as embedding.

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        in_id = tf.keras.layers.Input(shape=(self._time_steps,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self._time_steps,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self._time_steps,), name="segment_ids")
        inputs = [in_id, in_mask, in_segment]

        # Instantiate the custom Bert Layer
        x = Bert(pretrained_model_path=self._pretrained_model_path,
                 output_size=self._output_size,
                 n_layers_to_finetune=self._n_layers_to_finetune,
                 pooling=Bert.Pooling.ENCODER_OUT)(inputs)

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
