import tensorflow as tf
import numpy as np
from scipy.special import softmax


class BalancedTrainGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 inputs,
                 outputs,
                 batch_size: int = 32,
                 alpha_growth_start: int = 5,
                 alpha_growth_rate: float = 0.05,
                 alpha_growth_freq: int = 3,
                 max_alpha: float = 0.8
                 ):

        assert len(inputs) == len(outputs), "Length of inputs and outputs must be the same!"

        self._inputs = inputs
        self._outputs = outputs

        self._n_classes = len(np.unique(outputs))
        # start with equal probabilities
        self._probs = [1 / self._n_classes] * self._n_classes

        self._lut = np.empty([len(inputs)], dtype=[("index", int, 1),
                                                   ("class", int, 1),
                                                   ("prob", np.float, 1)])
        self._lut["index"] = np.arange(len(inputs))

        # if type(outputs[0]) == int:
        self._lut["class"] = outputs
        # else:
        #  self._lut["class"] = np.argmax(outputs, axis=1)

        self._lut["prob"] = softmax([1 / self._n_classes] * len(self._lut))

        self._alpha = 0

        if max_alpha > 1 or max_alpha < 0:
            raise ValueError("Maximum of alpha must be between 0 and 1.")
        self._max_alpha = max_alpha

        if alpha_growth_start < 0:
            raise ValueError("Start of alpha growth must be 0 or higher. (epoch count)")
        self._alpha_growth_start = alpha_growth_start

        if alpha_growth_rate <= 0:
            raise ValueError("Rate of alpha growth must be higher than 0.")
        self._alpha_growth_rate = alpha_growth_rate

        if alpha_growth_freq < 0:
            raise ValueError("Frequency of alpha growth must be higher than 0. (epoch count)")
        self._alpha_growth_freq = alpha_growth_freq

        self._batch_size = batch_size

        self._epoch_counter = 0

    # def __getitem__(self, index):
    #    return np.random.choice(inputs, self._batch_size, p=self._lut["prob"], replace=False)

    def __getitem__(self, index):
        chs = np.random.choice(np.arange(self._n_classes), self._batch_size, p=self._probs)

        unique, counts = np.unique(chs, return_counts=True)

        res = []

        for class_, count in zip(unique, counts):
            chs = np.random.choice(self._lut["index"][self._lut["class"] == class_], count, replace=False)

            res.extend(chs)

        np.random.shuffle(res)

        return self._inputs[res], self._outputs[res]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self._inputs) / self._batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        if self._epoch_counter == self._alpha_growth_start or self._epoch_counter > self._alpha_growth_start \
                and (self._epoch_counter - self._alpha_growth_start) % self._alpha_growth_freq == 0 \
                and (self._alpha + self._alpha_growth_rate) <= self._max_alpha:
            self._alpha += self._alpha_growth_rate

        np.random.shuffle(self._lut)

        self._epoch_counter += 1

    def update_probs(self, errors_per_class):
        assert len(errors_per_class) == len(self._probs)

        # convert them so the less accurate has the biggest probability
        # also scale from (0,1) to (0,10) to have a bigger weight on less accurate
        # classes_accuracies = 10 - (classes_accuracies*10)

        self._probs = self._alpha * errors_per_class + (1 - self._alpha) * self._probs
        # self._probs = errors_per_class

        # self._lut["prob"] = softmax([self._probs[c] for c in self._lut["class"]])
