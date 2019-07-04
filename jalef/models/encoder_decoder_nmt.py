from tensorflow.python.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed
from tensorflow.python.keras.models import Model

from .nlp_model import NLPModel
from jalef.layers import AttentionBlock


class EncoderDecoderNMT(NLPModel):

    def __init__(self,
                 target_vocab_size=20000,
                 use_attention=True,
                 use_shared_attention_vector=True,
                 bidirectional_encoder=True,
                 embedding_matrix=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._target_vocab_size = target_vocab_size
        self._use_attention = use_attention
        self._use_shared_attention_vector = use_shared_attention_vector
        self._bidirectional_encoder = bidirectional_encoder
        self._embedding_matrix = embedding_matrix

    def _create_model(self):
        inputs = Input(shape=(self._time_steps,))

        if self._embedding_matrix is None:
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, trainable=True)(inputs)
        else:
            x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, weights=[self._embedding_matrix],
                          trainable=self._trainable_embeddings)(inputs)

        for i in range(self._n_lstm_layers):
            if self._bidirectional_encoder:
                x = Bidirectional(LSTM(units=self._lstm_units_size[i], return_sequences=True,
                                       dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate))(x)
            else:
                x = LSTM(units=self._lstm_units_size[i], return_sequences=True,
                         dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate)(x)

        if self._use_attention:
            x = AttentionBlock(use_shared_attention_vector=self._use_shared_attention_vector, name='attention')(x)

        for i in range(self._n_lstm_layers)[::-1]:
            n_units = self._lstm_units_size[i]

            if self._bidirectional_encoder:
                n_units *= 2  # outputs are concatenated

            x = LSTM(n_units, return_sequences=True)(x)

        outputs = TimeDistributed(Dense(units=self._target_vocab_size, activation='softmax'))(x)

        self._model = Model(inputs, outputs)
