from keras.regularizers import Regularizer
import keras.backend as K
import tensorflow as tf

def svd(tensor):
    import numpy as np
    tensor = tensor.reshape(-1, tensor.shape[-1])
    _, s, _ = np.linalg.svd(tensor, full_matrices=True)
    return s.sum()

class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """
    
    def __init__(self, l1=0., l2=0., alpha=1.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.alpha = alpha
    
    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
            _, s, _ = tf.svd(tensor=x, full_matrices=True)
            s = tf.reduce_sum(s)
            regularization += self.alpha * s
        return regularization
    
    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}

# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)

def l2(l=0.01):
    return L1L2(l2=l)

def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
