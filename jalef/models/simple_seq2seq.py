from typing import Union, List
import numpy as np

from tensorflow.python.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed, Concatenate
from tensorflow.python.keras.models import Model

from .engine import Seq2SeqCore
from jalef.layers.attention import AttentionBlock


class SimpleSeq2Seq(Seq2SeqCore):
    """A simple Seq2Seq architecture using Word2Vec word embeddings.

    The encoder can be set to use bidirectional LSTM.
    """

    def __init__(self,
                 n_lstm_units: int,
                 dropout_rate: float,
                 recurrent_dropout_rate: float,
                 **kwargs):

        super().__init__(**kwargs)

        self._n_lstm_units = n_lstm_units
        self._dropout_rate = dropout_rate
        self._recurrent_dropout_rate = recurrent_dropout_rate

        self._encoder_inputs = None
        self._encoder_states = None
        self._encoder_outputs = None

        self._decoder_inputs = None
        self._decoder_embeddings = None
        self._decoder_lstm = None
        self._decoder_attention = None
        self._decoder_dense = None
        self._decoder_outputs = None

    def _construct_train_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model used at training.

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        # Encoder

        self._encoder_inputs = Input(shape=(self._time_steps,), name='encoder_inputs')

        if self._source_embedding_matrix is None:
            encoder_embedding = \
                Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim,
                          trainable=self._trainable_embeddings, name='encoder_embeddings')(self._encoder_inputs)
        else:
            encoder_embedding = \
                Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim,
                          weights=[self._source_embedding_matrix],
                          trainable=self._trainable_embeddings, name='encoder_embeddings')(self._encoder_inputs)

        if self._bidirectional_encoder:
            self._encoder_outputs, forward_h, forward_c, backward_h, backward_c = \
                Bidirectional(layer=LSTM(units=self._n_lstm_units, return_sequences=True, return_state=True,
                                         dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate),
                              name='encoder_LSTM')(encoder_embedding)

            encoder_h = Concatenate()([forward_h, backward_h])
            encoder_c = Concatenate()([forward_c, backward_c])
        else:
            self._encoder_outputs, encoder_h, encoder_c = \
                LSTM(units=self._n_lstm_units, return_sequences=True, return_state=True, name='encoder_LSTM',
                     dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate)(encoder_embedding)

        self._encoder_states = [encoder_h, encoder_c]

        # Decoder

        self._decoder_inputs = Input(shape=(self._time_steps,), name='decoder_inputs')

        if self._target_embedding_matrix is None:
            self._decoder_embeddings = \
                Embedding(input_dim=self._target_vocab_size, output_dim=self._embedding_dim,
                          trainable=self._trainable_embeddings, name='decoder_embeddings')(self._decoder_inputs)
        else:
            self._decoder_embeddings = \
                Embedding(input_dim=self._target_vocab_size, output_dim=self._embedding_dim,
                          trainable=self._trainable_embeddings, weights=[self._target_embedding_matrix],
                          name='decoder_embeddings')(self._decoder_inputs)

        n_units = self._n_lstm_units

        if self._bidirectional_encoder:
            n_units *= 2  # outputs are concatenated

        self._decoder_lstm = LSTM(units=n_units, return_sequences=True, return_state=True, dropout=self._dropout_rate,
                                  recurrent_dropout=self._recurrent_dropout_rate, name='decoder_LSTM')

        self._decoder_outputs, _, _ = self._decoder_lstm(inputs=self._decoder_embeddings,
                                                         initial_state=self._encoder_states)

        if self._use_attention:
            self._decoder_attention = AttentionBlock(use_shared_attention_vector=self._use_shared_attention_vector,
                                                     name='decoder_attention')

            self._decoder_outputs = self._decoder_attention(self._decoder_outputs)

        self._decoder_dense = TimeDistributed(layer=Dense(units=self._target_vocab_size, activation='softmax'))

        self._decoder_outputs = self._decoder_dense(self._decoder_outputs)

        self._model = Model(inputs=[self._encoder_inputs, self._decoder_inputs], outputs=[self._decoder_outputs])

        if print_summary:
            self._model.summary()

    def _construct_inference_model(self, print_summary: bool, **kwargs) -> None:
        """Construct the model used at inference (e.g. use in production).

        :param print_summary: Print model summary after compilation.
        :param kwargs: -
        :return: None
        """

        # Encoder

        self._encoder_inf_model = Model(inputs=self._encoder_inputs, outputs=self._encoder_states)

        # Decoder

        decoder_state_input_h = Input(shape=(self._n_lstm_units,), name='decoder_inf_input_h')
        decoder_state_input_c = Input(shape=(self._n_lstm_units,), name='decoder_inf_input_c')

        decoder_inf_states_input = [decoder_state_input_h, decoder_state_input_c]

        decoder_inf_output, decoder_inf_h, decoder_inf_c = self._decoder_lstm(inputs=self._decoder_embeddings,
                                                                              initial_state=decoder_inf_states_input)

        decoder_inf_states = [decoder_inf_h, decoder_inf_c]

        if self._use_attention:
            decoder_inf_output = self._decoder_attention(decoder_inf_output)

        decoder_inf_output = self._decoder_dense(decoder_inf_output)

        self._decoder_inf_model = Model(inputs=[self._decoder_inputs] + decoder_inf_states_input,
                                        outputs=[decoder_inf_output] + decoder_inf_states)

        if print_summary:
            print("Encoder model:")
            self._encoder_inf_model.summary()
            print("\nDecoder model:")
            self._decoder_inf_model.summary()

    def predict(self, X_test: Union[np.ndarray, List[np.ndarray]], save_predictions: bool = False,
                path: str = "") -> np.ndarray:

        # TODO: implement prediction
        raise NotImplementedError()
