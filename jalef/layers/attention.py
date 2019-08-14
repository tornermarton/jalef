from tensorflow.python.keras.backend import int_shape
import tensorflow as tf


class AttentionBlock(tf.keras.Model):

    def __init__(self, use_shared_attention_vector: bool=True, **kwargs):
        self.shared_attention_vector = use_shared_attention_vector
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        time_steps = int_shape(inputs)[1]
        input_dim = int_shape(inputs)[2]

        a = tf.keras.layers.Permute((2, 1))(inputs)
        attention_probs = tf.keras.layers.Dense(time_steps, activation='softmax', name='attention_probs')(a)

        if self.shared_attention_vector:
            dim_reduction = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name='dim_reduction')(attention_probs)
            attention_probs = tf.keras.layers.RepeatVector(input_dim)(dim_reduction)

        attention_mat = tf.keras.layers.Permute((2, 1), name='attention_matrix')(attention_probs)
        outputs = tf.keras.layers.Multiply(name='attention_mul')([inputs, attention_mat])

        return outputs