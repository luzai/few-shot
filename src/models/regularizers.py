from keras.regularizers import l2
from keras.regularizers import Regularizer
import keras.backend as K
import tensorflow as tf
import numpy as np


def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/-]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/-]+)(?:_\d+)?', name)[0]
    return name


def matrix_symmetric(x):
    return (x + tf.transpose(x, [0, 2, 1])) / 2


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])

    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res


@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    """
    Define the gradient for SVD
    References
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers

    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------
    """

    s, u, v = op.outputs
    # print grad_s.get_shape().as_list(), grad_u.get_shape().as_list(), grad_v.get_shape().as_list()
    # print s.get_shape().as_list(), u.get_shape().as_list(), v.get_shape().as_list()

    v_t = tf.transpose(v, [0, 2, 1])

    with tf.name_scope('K'):
        K = get_eigen_K(s, True)
    inner = matrix_symmetric(K * tf.matmul(v_t, grad_v))

    # Create the shape accordingly.
    u_shape = u.get_shape()[1].value
    v_shape = v.get_shape()[1].value

    # Recover the complete S matrices and its gradient
    eye_mat = tf.eye(v_shape, u_shape)
    realS = tf.matmul(tf.reshape(tf.matrix_diag(s), [-1, v_shape]), eye_mat)
    realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])

    real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(grad_s), [-1, v_shape]), eye_mat)
    real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])

    dxdz = tf.matmul(u, tf.matmul(2 * tf.matmul(realS, inner) + real_grad_S, v_t))
    return dxdz


def Svd(x):
    s, u, v = tf.svd(x, full_matrices=True)
    return s, u, v


class OrthoReg(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, alpha=1e-4, l2=1e-4, stop=False):
        self.stop = stop
        self.alpha = K.cast_to_floatx(alpha)
        print tf.get_default_graph()
        self.l2 = K.cast_to_floatx(l2)


    def __call__(self, x):
        regularization = 0.
        if self.alpha != 0:
            shape_x = x.get_shape().as_list()
            x_reshaped = tf.reshape(x, (np.prod(shape_x[:-1]), shape_x[-1]))
            x_reshaped = tf.expand_dims(x_reshaped, axis=0)

            shape_x = x_reshaped.get_shape().as_list()
            if shape_x[1] < shape_x[2]:
                x_reshaped = tf.transpose(x_reshaped, (0, 2, 1))

            s, u, v = Svd(x_reshaped)
            s_summ = tf.summary.histogram('single-value', s)
            tf.add_to_collection('s_summ',s_summ)
            s_min = tf.reduce_min(s)
            s = s - s_min
            reg_ = self.alpha * (
                tf.reduce_sum(tf.square(s)) +
                tf.reduce_sum(0. * u) +
                tf.reduce_sum(0. * v)
            )
            if self.stop:
                reg_ = tf.stop_gradient(reg_)
            regularization += reg_
        if self.l2 != 0:
            reg_ = K.sum(self.l2 * K.square(x))
            if self.stop:
                reg_ = tf.stop_gradient(reg_)
            regularization += reg_
        return regularization

    def get_config(self):
        return {'alpha': float(self.alpha),
                }


def ortho_l2_reg(alpha=1e-4, l=1e-4, stop=False):
    return OrthoReg(alpha=alpha, l2=l, stop=stop)
