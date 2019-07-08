from tensorflow.python.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, TimeDistributed
from tensorflow.python.keras.models import Model

from .engine import Seq2SeqCore, CustomLSTMModelCore
from jalef.layers import AttentionBlock


class EncoderDecoderNMT(Seq2SeqCore, CustomLSTMModelCore):

    def __init__(self,
                 dropout_rate,
                 recurrent_dropout_rate,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self._dropout_rate = dropout_rate
        self._recurrent_dropout_rate = recurrent_dropout_rate

    def _construct_train_model(self, print_summary):
        inputs = Input(shape=(self._time_steps,))

        if self._source_embedding_matrix is None:
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim, trainable=True)(inputs)
        else:
            x = Embedding(input_dim=self._source_vocab_size, output_dim=self._embedding_dim, weights=[self._source_embedding_matrix],
                          trainable=self._trainable_embeddings)(inputs)

        for i in range(len(self._lstm_units_size)):
            if self._bidirectional_encoder:
                x = Bidirectional(LSTM(units=self._lstm_units_size[i], return_sequences=True,
                                       dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate))(x)
            else:
                x = LSTM(units=self._lstm_units_size[i], return_sequences=True,
                         dropout=self._dropout_rate, recurrent_dropout=self._recurrent_dropout_rate)(x)

        if self._use_attention:
            x = AttentionBlock(use_shared_attention_vector=self._use_shared_attention_vector, name='attention')(x)

        for i in range(len(self._lstm_units_size))[::-1]:
            n_units = self._lstm_units_size[i]

            if self._bidirectional_encoder:
                n_units *= 2  # outputs are concatenated

            x = LSTM(n_units, return_sequences=True)(x)

        outputs = TimeDistributed(Dense(units=self._target_vocab_size, activation='softmax'))(x)

        self._model = Model(inputs, outputs)

    def _construct_inference_model(self, print_summary):
        # Here no inference model is needed
        pass

    def predict(self, X):
        # TODO: implement prediction
        raise NotImplementedError()

