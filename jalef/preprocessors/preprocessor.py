from abc import ABC, abstractmethod
from tensorflow.python.keras.utils import to_categorical
import numpy as np


class Preprocessor(ABC):

    def __init__(self, max_sequence_length):
        self._max_seq_len = max_sequence_length

    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        self.fit(texts=texts)

        return self.transform(texts=texts)

    def fit_transform_classification(self, texts, labels):
        return np.array([self.fit_transform(texts=texts), to_categorical(labels)],
                        dtype=[("input", np.ndarray, 1), ("output", np.ndarray, 1)])
