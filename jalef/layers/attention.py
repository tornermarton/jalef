from keras import backend as K
from keras.backend import int_shape
from keras.layers import Layer, Permute, Dense, Lambda, RepeatVector, Multiply


class AttentionBlock(Layer):

    def __init__(self, use_shared_attention_vector=True, **kwargs):
        self.shared_attention_vector = use_shared_attention_vector
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        time_steps = int_shape(inputs)[1]
        input_dim = int_shape(inputs)[2]

        a = Permute((2, 1))(inputs)
        attention_probs = Dense(time_steps, activation='softmax', name='attention_probs')(a)

        if self.shared_attention_vector:
            dim_reduction = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(attention_probs)
            attention_probs = RepeatVector(input_dim)(dim_reduction)

        attention_mat = Permute((2, 1), name='attention_matrix')(attention_probs)
        outputs = Multiply(name='attention_mul')([inputs, attention_mat])

        return outputs
