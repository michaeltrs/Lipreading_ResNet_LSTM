import numpy as np
import tensorflow as tf
from helper_functions import *
from tensorflow.contrib.slim.nets import resnet_v2


# Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(input, filter):
    """
    input: A Tensor. Must be one of the following types:
    half, bfloat16, float32, float64.
    Shape [batch, in_depth, in_height, in_width, in_channels]
    filter: A Tensor. Must have the same type as input.
    Shape [filter_depth, filter_height, filter_width, in_channels, out_channels].
    in_channels must match between input and filter.
    """
    return tf.nn.conv3d(input, filter,
                        strides=[1, 1, 2, 2, 1],
                        padding='SAME')


def batch_norm(input):
    # Batch normalization.
    return tf.contrib.layers.batch_norm(
                input,
                data_format='NHWC',  # Matching the "cnn" tensor shape
                center=True,
                scale=True,
                is_training=training,
                scope='cnn3d-batch_norm')


def max_pool3d(x):
    return tf.nn.max_pool3d(x,
                            strides=[1, 1, 2, 2, 1],
                            ksize=[1, 3, 3, 3, 1],
                            padding='SAME')


# USER INPUT - START ---------------------------------------------------#
width = 112
height = 112
depth = 1
num_frames = 28
n_labels = 512

# USER INPUT - END ---------------------------------------------------- #


sess = tf.InteractiveSession()

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

x = tf.placeholder(tf.float32, shape=[None, num_frames, width, height, 1])
print("shape of input is %s" % x.get_shape)

y = tf.placeholder(tf.float32, shape=[None, 1, 1, n_labels])

# First Convolution Layer - Image input, use 64 3D filters of size 5x5x5
# shape of weights is (dx, dy, dz, #in_filters, #out_filters)
# W_conv1 = weight_variable([5, 5, 5, 1, 64])
W_conv1 = weight_variable([5, 7, 7, 1, 64])

b_conv1 = bias_variable([64])

# check if there is a need to reshape x !
# x_image = tf.reshape(x, [-1,width,height,depth,1])

# apply first convolution
z_conv1 = conv3d(x, W_conv1) + b_conv1
# apply batch normalization
z_conv1_bn = batch_norm(z_conv1)
# apply relu activation
h_conv1 = tf.nn.relu(z_conv1_bn)
print("shape after 1st convolution is %s" % h_conv1.get_shape)

# apply max pooling
h_pool1 = max_pool3d(h_conv1)
print(h_pool1.get_shape)
print("shape after 1st pooling is %s" % h_pool1.get_shape)

# resnet model - need to change that to 34 layer model
features, end_points = resnet_v2.resnet_v2_50(
    h_pool1[:, 0, :, :, :], num_classes=512)



# Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=features))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(features, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for i in range(3):
    print(i)
    x_train = get_data()
    y_train = np.zeros(512).astype(np.float32)
    y_train[0] = 1.
    y_train = y_train.reshape(1, 1, 1, 512)
    # Logging every 100th iteration in the training process.
    if i%5 == 0:
        test_var1 = features.eval(feed_dict={x:x_train, y:y_train})
        # print(test_var1.shape)
        #print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: x_train, y: y_train}) #, keep_prob: 0.5})


# a = x_input[0,0,:, :, 0]#[0]
# b = a[:, :][0]
# np.savetxt("/homes/mat10/Programming/OpenCV/frames/test.csv", a)
# a = test_var1[0][0][0]
# test_var1[:, 0, :, :, :]
