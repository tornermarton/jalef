import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
import numpy as np
from keras.utils import to_categorical

from jalef.preprocessors.preprocessor import Preprocessor


class BertPreprocessor(Preprocessor):
    """
    This class can be used to do all the work to create the input and output for a Neural Network using BERT
    as embedding.

    Input - list of text sequences
    Output - list of network inputs and outputs (one hot encoded)
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        info = hub.Module(spec=self._pretrained_model_path)(signature="tokenization_info", as_dict=True)

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

    def _tokenize_example(self, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        input_ids = [0] * self._max_seq_len
        input_mask = [0] * self._max_seq_len
        input_segment_ids = [0] * self._max_seq_len

        # Padding contains only zeros
        if isinstance(example, BertPreprocessor.PaddingInputExample):
            return input_ids, input_mask, input_segment_ids, 0

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
        for i in range(len(input_ids)):
            input_mask[i] = 1

        # safety check
        assert len(input_ids) == self._max_seq_len
        assert len(input_mask) == self._max_seq_len
        assert len(input_segment_ids) == self._max_seq_len

        return input_ids, input_mask, input_segment_ids, example.label

    def run(self, texts, labels):
        examples = []

        for text, label in zip(texts, labels):
            examples.append(
                Preprocessor.InputExample(guid=None, text=" ".join(text), label=label)
            )

        input_ids, input_masks, segment_ids, labels = [], [], [], []

        for example in examples:
            input_id, input_mask, segment_id, label = self._tokenize_example(example)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)

        return (
            [np.array(input_ids), np.array(input_masks), np.array(segment_ids)],
            to_categorical(labels),
        )
