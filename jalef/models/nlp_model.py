from .core import Core


class NLPModel(Core):

    def __init__(self,
                 time_steps=50,
                 vocab_size=20000,
                 embedding_dim=300,
                 trainable_embeddings=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._trainable_embeddings = trainable_embeddings
