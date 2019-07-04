from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from abc import ABC, abstractmethod


class Core(ABC):
    """
    The base for all models in this package.
    """

    def __init__(self,
                 lstm_units_size=256,
                 hidden_units_size=256,
                 dropout_rate=0.1,
                 recurrent_dropout_rate=0.1,
                 optimizer='adam',
                 loss=None,
                 metrics=None,
                 epochs=10,
                 batch_size=128,
                 shuffle=True,
                 patience=5,
                 min_delta=0.005,
                 weights_file='weights.hdf5',
                 log_file='history.csv'
                 ):

        if type(lstm_units_size) is list:
            self._lstm_units_size = lstm_units_size
        else:
            self._lstm_units_size = [lstm_units_size]

        self._n_lstm_layers = len(self._lstm_units_size)

        if type(hidden_units_size) is list:
            self._hidden_units_size = hidden_units_size
        else:
            self._hidden_units_size = [hidden_units_size]

        self._n_hidden_layers = len(self._hidden_units_size)

        if self._n_lstm_layers < 1 or self._n_hidden_layers < 1:
            raise ValueError(
                "The number of layers can't be less than 1. " +
                "(Architectures without this kind of layer ignore this parameter)")

        self._dropout_rate = dropout_rate
        self._recurrent_dropout_rate = recurrent_dropout_rate
        self._optimizer = optimizer
        self._loss = loss

        if metrics is None:
            metrics = ['acc']
        self._metrics = metrics

        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._patience = patience
        self._min_delta = min_delta
        self._weights_file = weights_file
        self._log_file = log_file

        self._model = None

    @abstractmethod
    def _create_model(self):
        pass

    def compile(self, print_summary=False):
        self._create_model()

        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

        if print_summary:
            self._model.summary()

    def train(self, X_train, y_train, X_valid, y_valid):
        model_checkpoint = ModelCheckpoint(filepath=self._weights_file, monitor='val_loss', verbose=1,
                                           save_best_only=True)
        early_stopping = EarlyStopping(patience=self._patience, min_delta=self._min_delta)
        csv_logger = CSVLogger(self._log_file, append=True, separator=';')

        self._model.fit(X_train, y_train,
                        batch_size=self._batch_size,
                        epochs=self._epochs,
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        shuffle=self._shuffle,
                        callbacks=[model_checkpoint, early_stopping, csv_logger]
                        )
