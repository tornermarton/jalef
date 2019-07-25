import gensim
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from jalef.preprocessing.preprocessor import Preprocessor


class Word2VecPreprocessor(Preprocessor):
    """
    This class can be used to do all the work to create the input and output for a Neural Network using Word2Vec
    as embedding.

    Input - list of text sequences
    Output - list of network inputs and outputs (one hot encoded)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ',)

    def fit(self, texts):
        self._tokenizer.fit_on_texts(texts=texts)

    def transform(self, texts):
        sequences = self._tokenizer.texts_to_sequences(texts=texts)

        sequences = pad_sequences(sequences=sequences, maxlen=self._max_seq_len, padding='post', truncating='post')

        return np.array(sequences)

    def inverse_transform(self, sequences):
        # TODO: implement Word2VecPreprocessor inverse_transform
        raise NotImplementedError

    def get_embedding_matrix(self, embedding_dimension, pretrained_model_path):
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
