from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed
from keras.models import Model

from .core import Core
from jalef.layers import AttentionBlock


class EncoderDecoderNMT(Core):

    def __init__(self,
                 target_vocab_size=20000,
                 use_attention=True,
                 use_shared_attention_vector=True,
                 bidirectional_encoder=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._target_vocab_size = target_vocab_size
        self._use_shared_attention_vector = use_shared_attention_vector
        self._bidirectional_encoder = bidirectional_encoder

    def compile(self, embedding_matrix=None, **kwargs):
        inputs = Input(shape=(self._time_steps,))

        if embedding_matrix is None:
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(self._vocab_size, self._embedding_dim, trainable=True)(inputs)
        else:
            x = Embedding(self._vocab_size, self._embedding_dim, weights=[embedding_matrix],
                          trainable=self._trainable_embeddings)(inputs)

        for i in range(self._n_lstm_layers):
            if self._bidirectional_encoder:
                x = Bidirectional(LSTM(self._lstm_units_size[i], return_sequences=True))(x)
            else:
                x = LSTM(self._lstm_units_size[i], return_sequences=True)(x)

        x = AttentionBlock(use_shared_attention_vector=self._use_shared_attention_vector, name='attention')(x)

        for i in range(self._n_lstm_layers)[::-1]:
            n_units = self._lstm_units_size[i]

            if self._bidirectional_encoder:
                n_units *= 2  # outputs are concatenated

            x = LSTM(n_units, return_sequences=True)(x)

        outputs = TimeDistributed(Dense(self._target_vocab_size, activation='softmax'))(x)

        self._model = Model(inputs, outputs)

        self._compile(**kwargs)
