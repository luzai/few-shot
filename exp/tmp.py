import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]

# def slice_batch(x, n_gpus, part):
#     sh = K.shape(x)
#     L = sh[-1] / n_gpus
#     if part == n_gpus - 1:
#         return x[:, :, part*L]
#     return x[:, :, part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape,
                             arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))
    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)
    return Model(input=[x], output=merged)
