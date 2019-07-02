import tensorflow as tf
import tensorflow_hub as hub


class Bert(tf.keras.layers.Layer):
    """
    This is a custom keras layer integrating Bert from tf-hub.

    Source: https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
    Bert: https://arxiv.org/pdf/1810.04805.pdf

    Search for pretrained models: https://tfhub.dev/
    (Here we use the actual best as default - Whole Word Masking, uncased version)
    """

    SUPPORTED_POOLING_TYPES = ["first", "reduce_mean"]

    def __init__(self,
                 pretrained_model_path,
                 output_size,
                 n_layers_to_finetune=0,
                 pooling="reduce_mean",
                 **kwargs):

        self._pretrained_model_path = pretrained_model_path
        # This should be set according to the used model (H-XXXX)
        self._output_size = output_size

        self._trainable = n_layers_to_finetune != 0
        self._n_layers_to_finetune = n_layers_to_finetune

        if pooling not in Bert.SUPPORTED_POOLING_TYPES:
            raise NameError(
                "Unsupported pooling type {}! Please use one from Bert.SUPPORTED_POOLING_TYPES.".format(pooling)
            )

        self._pooling = pooling

        self._bert_module = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        self._bert_module = hub.Module(
            spec=self._pretrained_model_path, trainable=self._trainable, name="{}_module".format(self.name)
        )

        # Remove unused layers
        trainable_vars = self._bert_module.variables
        if self._pooling == "first":
            trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]
            trainable_layers = ["pooler/dense"]

        elif self._pooling == "reduce_mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                "Unsupported pooling type {}! Please use one from Bert.SUPPORTED_POOLING_TYPES.".format(self._pooling)
            )

        # Select how many layers to fine tune
        for i in range(self._n_layers_to_finetune):
            trainable_layers.append("encoder/layer_{}".format(str(23 - i)))

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self._bert_module.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self._pooling == "first":
            pooled = self._bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self._pooling == "reduce_mean":
            result = self._bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            def mul_mask(x, m):
                return x * tf.expand_dims(m, axis=-1)

            def masked_reduce_mean(x, m):
                return tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(
                "Unsupported pooling type {}! Please use one from Bert.SUPPORTED_POOLING_TYPES.".format(self._pooling)
            )

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size
