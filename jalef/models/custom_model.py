from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.python.keras.models import Model

from jalef.preprocessing import Word2VecPreprocessor
from jalef.layers import Bert


class CustomModel():
    """
    Fully customizable basic model. Task can be classification or regression. If classification the number of neurons
    in the output layer will be equal to the num_classes parameter, if regression than 1.
    """

    SUPPORTED_EMBEDDINGS = ["bert", "word2vec"]

    def __init__(self,
                 task="classification",
                 num_classes=2,
                 use_pretrained_embeddings=False,
                 embedding_type=None,
                 embedding_model_path=None,
                 lstm_units_size=256,
                 use_bidirectional_lstm=False,
                 hidden_units_size=256,
                 **kwargs):
        super().__init__(**kwargs)

        if task != "classification" and task != "regression":
            raise ValueError("Supported tasks are classification and regression!")

        self._task = task
        self._num_classes = num_classes

        if use_pretrained_embeddings and (embedding_type is None or embedding_model_path is None):
            raise ValueError("If you use pretrained embeddings please specify the type and the model path!")

        self._use_pretrained_embeddings = use_pretrained_embeddings

        if embedding_type not in CustomModel.SUPPORTED_EMBEDDINGS:
            raise ValueError("Please use one type from CustomModel.SUPPORTED_EMBEDDINGS!")

        self._embedding_type = embedding_type
        self._embedding_model_path = embedding_model_path

        self._use_bidirectional_lstm = use_bidirectional_lstm

    def _create_model(self):
        if not self._use_pretrained_embeddings:
            inputs = Input(shape=(self._time_steps,))
            # Assert that weights are trainable, because otherwise doesn't make sense
            x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, trainable=True)(inputs)
        else:
            if self._embedding_type == "bert":
                pass
            elif self._embedding_type == "word2vec":
                wp = Word2VecPreprocessor(max_sequence_length=self._time_steps, pretrained_model_path=self._embedding_model_path)

                embedding_matrix = wp.get_embedding_matrix(self._embedding_dim)

                inputs = Input(shape=(self._time_steps,))

                x = Embedding(input_dim=self._vocab_size, output_dim=self._embedding_dim, weights=[embedding_matrix],
                              trainable=self._trainable_embeddings)(inputs)
            else:
                raise ValueError("Please use one type from CustomModel.SUPPORTED_EMBEDDINGS!")

        for i in range(self._n_lstm_layers):
            if self._use_bidirectional_lstm:
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
