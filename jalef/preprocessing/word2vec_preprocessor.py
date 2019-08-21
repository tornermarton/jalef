import gensim
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from jalef.preprocessing.preprocessor import Preprocessor


class Word2VecPreprocessor(Preprocessor):
    """Preprocessor for Neural Networks using Word2Vec word embedding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ', )

    def fit(self, texts):
        """Fit the tokenizer on the vocabulary of the given documents.

        :param texts: The sequences of words
        :return: self
        """
        self._tokenizer.fit_on_texts(texts=texts)

        return self

    def transform(self, texts):
        """Transform the sequences of words into sequences of tokens.

        :param texts: The sequences of words.
        :return: The sequences of tokens.
        """

        sequences = self._tokenizer.texts_to_sequences(texts=texts)

        sequences = pad_sequences(sequences=sequences, maxlen=self._max_seq_len, padding='post', truncating='post')

        return np.array(sequences)

    def inverse_transform(self, sequences):
        """Transform sequences of tokens back to sequences of words (sentences).

        :param sequences: The sequences of tokens.
        :return: The sequences of words
        """

        return self._tokenizer.sequences_to_texts(sequences=sequences)

    def get_embedding_matrix(self, embedding_dimension, pretrained_model_path):
        """Get the embedding matrix of the given pretrained model.

        Tokenizer must be fit before calling this method!

        :param embedding_dimension: The dimension of the embedding vectors (e.g. 300).
        :param pretrained_model_path: The full path to the pretrained model.
        :return: The embedding matrix.
        """

        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=pretrained_model_path, binary=True)

        word_index = self._tokenizer.word_index

        vocabulary_size = len(word_index) + 1

        embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))

        for word, i in word_index.items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                # word not in vocabulary -> assign random vector
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dimension)

        del word_vectors

        return embedding_matrix
