from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.python.keras.models import Model

from .nlp_model import NLPModel


class CustomModel(NLPModel):
    """
    Fully customizable basic model. Task can be classification or regression. If classification the number of neurons
    in the output layer will be equal to the num_classes parameter, if regression than 1.
    """

    def __init__(self,
                 task="classification",
                 num_classes=2,
                 bidirectional_lstm=True,
                 **kwargs):
        super().__init__(**kwargs)

        if task != "classification" and task != "regression":
            raise ValueError("Supported tasks are classification and regression!")

        self._task = task
        self._num_classes = num_classes
        self._bidirectional_lstm = bidirectional_lstm

    def compile(self, embedding_matrix=None, **kwargs):
        inputs = Input(shape=(self._time_steps,))

        if embedding_matrix is None:
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, trainable=True)(inputs)
        else:
            x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, weights=[embedding_matrix],
                          trainable=self._trainable_embeddings)(inputs)

        for i in range(self._n_lstm_layers):
            if self._bidirectional_lstm:
                x = Bidirectional(
                    LSTM(units=self._lstm_units_size[i], return_sequences=True, dropout=self._dropout_rate,
                         recurrent_dropout=self._recurrent_dropout_rate))(x)
            else:
                x = LSTM(units=self._lstm_units_size[i], return_sequences=True, dropout=self._dropout_rate,
                         recurrent_dropout=self._recurrent_dropout_rate)(x)

        for i in range(self._n_hidden_layers):
            x = Dense(units=self._hidden_units_size[i], activation='relu')(x)
            x = Dropout(rate=self._dropout_rate)(x)

        if self._task == "classification":
            outputs = Dense(units=self._num_classes, activation='softmax')(x)
        elif self._task == "regression":
            outputs = Dense(units=1, activation='sigmoid')
        else:
            raise ValueError("Supported tasks are classification and regression!")

        self._model = Model(inputs=inputs, outputs=outputs)

        self._compile(**kwargs)
