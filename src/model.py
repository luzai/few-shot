import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2


def resnet101(images, classes=10):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=1e-6)):
        logits, end_points = resnet_v2.resnet_v2_101(images, classes, is_training=True)

        logits = logits[:, 0, 0, :]

    return logits, end_points

def resnet50(images, classes=10):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(images, classes, is_training=True)

        logits = logits[:, 0, 0, :]

    return logits, end_points


import tensorflow.contrib.layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers

def resnet101_2(images, classes):
    # tf.reset_default_graph()
    # with tf.variable_scope('',reuse=True):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        # tf.get_variable_scope().reuse_variables()
        logits, end_points = resnet_v2.resnet_v2_101(images, global_pool=False, is_training=True)
        # logits = layers.batch_norm(
        #     logits, activation_fn=tf.nn.relu, scope='postnorm2')
        # logits = layers_lib.conv2d(logits, classes, [1, 1], padding='valid', scope='logits',
        #                            normalizer_fn=None,
        #                            activation_fn=None)
        # logits = tf.reduce_mean(logits, axis=1)
        # logits = tf.reduce_mean(logits, axis=1)
        logits = slim.max_pool2d(logits, (3, 3), stride=4)
        logits = slim.flatten(logits)
        logits = slim.fully_connected(logits, classes, scope='logits', activation_fn=None, normalizer_fn=None)
    return logits
