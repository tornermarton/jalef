from abc import ABC, abstractmethod


class Preprocessor(ABC):
    class PaddingInputExample(object):
        """Fake example so the num input examples is a multiple of the batch size.
      When running eval/predict on the TPU, we need to pad the number of examples
      to be a multiple of the batch size, because the TPU requires a fixed batch
      size. The alternative is to drop the last batch, which is bad because it means
      the entire output data won't be generated.
      We use this class instead of `None` because treating `None` as padding
      battches could cause silent errors.
      """

    class InputExample(object):
        """A single training/test example for simple sequence classification."""

        def __init__(self, guid, text, label=None):
            """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the sequence. This project only includes single sequence problems
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
            self.guid = guid
            self.text = text
            self.label = label

    def __init__(self, max_sequence_length, pretrained_model_path):
        self._max_seq_len = max_sequence_length
        self._pretrained_model_path = pretrained_model_path

    @abstractmethod
    def run(self, texts, labels):
        pass
