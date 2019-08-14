from typing import List

from tensorflow.python.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed
from tensorflow.python.keras.models import Model

from .engine import Seq2SeqCore
from jalef.layers import AttentionBlock


class EncoderDecoderNMT(Seq2SeqCore):

    """A simple Seq2Seq architecture implemented as an Encoder-Decoder model using Word2Vec word embeddings."""

    def __init__(self,                 
                 lstm_layer_sizes: List[int],
                 dropout_rate: float,
                 recurrent_dropout_rate: float,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        if type(lstm_layer_sizes) is not list:
            lstm_layer_sizes = [lstm_layer_sizes]

        self._lstm_layer_sizes: List[int] = lstm_layer_sizes
        
        self._dropout_rate = dropout_rate
        self._recurrent_dropout_rate = recurrent_dropout_rate

    def _construct_train_model(self, print_summary: bool, **kwargs) -> None:
        inputs = Input(shape=(self._time_steps,))

        if self._source_embedding_matrix is None:
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim, trainable=True)(inputs)
        else:
            x = Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim,
                          weights=[self._source_embedding_matrix],
                          trainable=self._trainable_embeddings)(inputs)

        for i in range(len(self._lstm_layer_sizes)):
            if self._bidirectional_encoder:
                x = Bidirectional(layer=LSTM(units=self._lstm_layer_sizes[i], return_sequences=True,
                                             dropout=self._dropout_rate,
                                             recurrent_dropout=self._recurrent_dropout_rate))(x)
            else:
                x = LSTM(units=self._lstm_layer_sizes[i], return_sequences=True,
                         dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate)(x)

        if self._use_attention:
            x = AttentionBlock(use_shared_attention_vector=self._use_shared_attention_vector, name='attention')(x)

        for i in range(len(self._lstm_layer_sizes))[::-1]:
            n_units = self._lstm_layer_sizes[i]

            if self._bidirectional_encoder:
                n_units *= 2  # outputs are concatenated

            x = LSTM(units=n_units, return_sequences=True)(x)

        outputs = TimeDistributed(layer=Dense(units=self._target_vocab_size, activation='softmax'))(x)

        self._model = Model(inputs=inputs, outputs=outputs)

    def _construct_inference_model(self, print_summary: bool, **kwargs):
        # Here no inference model is needed
        pass
