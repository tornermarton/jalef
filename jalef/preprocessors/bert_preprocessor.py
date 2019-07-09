import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
import numpy as np

from jalef.preprocessors.preprocessor import Preprocessor


class BertPreprocessor(Preprocessor):
    """
    This class can be used to do all the work to create the input and output for a Neural Network using BERT
    as embedding.

    Input - list of text sequences
    Output - list of network inputs and outputs (one hot encoded)
    """

    def __init__(self,
                 pretrained_model_path,
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

        self._tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        basic_tokens = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        self._CLS_token = basic_tokens[0]
        self._SEP_token = basic_tokens[1]

    def _padding_sentence(self):
        # this function returns a zero length sentence to pad last batch
        return [0] * self._max_seq_len, [0] * self._max_seq_len, [0] * self._max_seq_len

    def tokenize(self, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        input_ids = [0] * self._max_seq_len
        input_mask = [0] * self._max_seq_len
        input_segment_ids = [0] * self._max_seq_len

        tokens_input = self._tokenizer.tokenize(example.text)

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

    def fit(self, texts):
        # this function does nothing in case of Bert embedding but must be implemented
        pass

    def transform(self, texts):
        input_ids, input_masks, segment_ids = [], [], []

        for text in texts:
            input_id, input_mask, segment_id = self.tokenize(text)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        return np.array([np.array(input_ids), np.array(input_masks), np.array(segment_ids)],
                        dtype=[("input_ids", np.ndarray, 1), ("input_masks", np.ndarray, 1),
                               ("segment_ids", np.ndarray, 1)])
