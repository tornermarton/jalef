from typing import List

import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
import numpy as np

from jalef.preprocessing.preprocessor import Preprocessor


class BertPreprocessor(Preprocessor):
    """Preprocessor for BERT embedding.

    This class can be used to do all the work to create the inputs (and outputs) of a Neural Network using BERT
    as embedding. Currently only single sequence classification is supported.
    """

    def __init__(self,
                 pretrained_model_path: str,
                 **kwargs):

        super().__init__(**kwargs)

        info = hub.Module(spec=pretrained_model_path)(signature="tokenization_info", as_dict=True)

        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [
                    info["vocab_file"],
                    info["do_lower_case"]
                ]
            )

        # Create the tokenizer with the vocabulary of the pretrained model
        self._tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        basic_tokens = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        self._CLS_token = basic_tokens[0]
        self._SEP_token = basic_tokens[1]

    def _padding_sentence(self):
        """Return a zero length sentence to pad last batch.

        :return: Three sequences of zeros (tokens, masks, segment ids).
        """

        return [0] * self._max_seq_len, [0] * self._max_seq_len, [0] * self._max_seq_len

    def tokenize(self, text: str):
        """Convert a sequence of words into a sequence of tokens and also compute the masking- and segment ids.

        For further details please read BERT paper.

        :param text: The sequence of words.
        :return: The sequence of tokens, masks and segment ids.
        """

        input_ids = [0] * self._max_seq_len
        input_mask = [0] * self._max_seq_len
        input_segment_ids = [0] * self._max_seq_len

        tokens_input = self._tokenizer.tokenize(text)

        # if too long cut to size (the first token will be [CLS], the last [SEP])
        if len(tokens_input) > self._max_seq_len - 2:
            tokens_input = tokens_input[0: (self._max_seq_len - 2)]

        idx = 0
        input_ids[idx] = self._CLS_token
        idx += 1

        for element in self._tokenizer.convert_tokens_to_ids(tokens_input):
            input_ids[idx] = element
            idx += 1

        input_ids[idx] = self._SEP_token

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        for i in range(idx+1):
            input_mask[i] = 1

        # safety check
        assert len(input_ids) == self._max_seq_len
        assert len(input_mask) == self._max_seq_len
        assert len(input_segment_ids) == self._max_seq_len

        return input_ids, input_mask, input_segment_ids

    def fit(self, texts: List[str]) -> 'BertPreprocessor':
        """This function does nothing in case of BERT but must be implemented.

        :param texts: -
        :return: self
        """

        return self

    def transform(self, texts: List[str]) -> list:
        """Transform sequences of words into sequences of tokens, masks and segment ids.

        Masks are used to separate valid and padding tokens. Here the segment ids are always one since the whole
        sequence belongs together.

        For further details please read BERT paper.

        :param texts: The sequences of texts.
        :return: The sequences of tokens, masks and segment ids.
        """

        input_ids, input_masks, segment_ids = [], [], []

        for i, text in enumerate(texts):
            input_id, input_mask, segment_id = self.tokenize(text=text)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        return [np.array(input_ids), np.array(input_masks), np.array(segment_ids)]

    def inverse_transform(self, sequences: np.ndarray):
        """Transform sequences of tokens back to sequences of words (sentences).

        :param sequences: The sequences of tokens.
        :return: The sequences of words
        """

        return self._tokenizer.convert_ids_to_tokens(sequences)
