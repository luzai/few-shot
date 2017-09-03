from keras.layers import Layer


class Orthogonalization(Layer):
    def __init__(self, **kwargs):
        super(Orthogonalization, self).__init__(**kwargs)
        alpha=None