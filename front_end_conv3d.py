import tensorflow as tf
import numpy as np

FC_SIZE = 1024
DTYPE = tf.float32


def _weight_variable(name, shape):
    return tf.get_variable(name=name,
                           shape=shape, 
                           dtype=DTYPE, 
                           initializer=tf.truncated_normal_initializer(stddev=0.1))


# Why initialize bias to 0.1?
def _bias_variable(name, shape):
    return tf.get_variable(name=name,
                           shape=shape,
                           dtype=DTYPE,
                           initializer=tf.constant_initializer(0.1, dtype=DTYPE))


def conv3_front_end(input_, in_filters=16, out_filters=16):
    """
    3D Spatiotemporal convolution front end
    """
    with tf.variable_scope('conv1') as scope:

        filter_ = _weight_variable(name='weights',
                                   shape=[5, 5, 5, in_filters, out_filters])

        conv = tf.nn.conv3d(input=input_,
                            filter=filter_,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME')

        biases = _bias_variable('biases', [out_filters])

        bias = tf.nn.bias_add(conv, biases)

        conv1 = tf.nn.relu(bias, name=scope.name)

    pool1 = tf.nn.max_pool3d(conv1,
                             ksize=[1, 3, 3, 3, 1],
                             strides=[1, 2, 2, 2, 1],
                             padding='SAME')

    return pool1
