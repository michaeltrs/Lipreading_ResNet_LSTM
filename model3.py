import numpy as np
import tensorflow as tf
from helper_functions import *
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.rnn import BasicLSTMCell
# from bidirectional_lstm import Bi_LSTM_cell


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
n_labels = 256

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



# define batch size

# Split the 3d convolution out to feed it to resnet
# resnet_input_series = tf.split(axis=1, num_or_size_splits=28, value = h_pool1)
# resnet_input_series = tf.squeeze(resnet_input_series)
# resnet_input_series[0][:,0,:,:,:]

batch_size, seq_length, height, width, channels = h_pool1.get_shape().as_list()

h_pool1 = tf.reshape(h_pool1, (-1, height, width, channels))



# resnet model - need to change that to 34 layer model
features, end_points = resnet_v2.resnet_v2_50(
    h_pool1, num_classes=512)
print("shape after resnet is %s" % features.get_shape)


# linear layer after resnet 512 -> 256 (check this is correct)
dense1 = tf.layers.dense(inputs=features, units=256, activation=tf.nn.relu)
print("shape after linear layer is %s" % dense1.get_shape)

dense1 = tf.reshape(dense1, (-1, 256))
dense1_shape = dense1.get_shape().as_list()
dense1 = tf.reshape(dense1, (-1, seq_length, dense1_shape[1]))

# 2-layer Bidirectional LSTM
# predict and backpropagate at every time step
# lstm1 = Bi_LSTM_cell(input_size=256, hidden_layer_size=256, target_size=256)
# outputs = rnn.get_outputs()
# print("shape after single layer biLSTM is %s" % lstm1.get_shape)
# # Getting first output through indexing
# last_output = outputs[-1]
# # As rnn model output the final layer through Relu activation softmax is
# # used for final output.
# output = tf.nn.softmax(last_output)

# lstm = BasicLSTMCell(256)
# # Initial state of the LSTM memory.
# initial_state = state = tf.zeros([1, 256])
# # for one step (change that)
# output, state = lstm(dense1, state)

batch_size, seq_length, num_features = dense1.get_shape().as_list()


# tf squeeze

lstm_input = dense1#[:, 0, :, :] #tf.reshape(dense1, (batch_size, 1, num_features))

cell = tf.nn.rnn_cell.LSTMCell(num_features, state_is_tuple=True)
lstm_out, _ = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32)

# get only the last output from lstm
lstm_out = tf.transpose(lstm_out, [1, 0, 2])
last = tf.gather(lstm_out, int(lstm_out.get_shape()[0]) - 1)
print("shape of lstm output is %s" % last.get_shape)

pred = tf.nn.softmax(last)




# Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
test_var = pred #dense1
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=test_var))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(test_var, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for epoch in range(3):
    print(epoch)


    x_train = get_data()
    y_train = np.array(x_train.shape[0] * [np.zeros(n_labels).astype(np.float32)])
    for i in range(x_train.shape[0]):
        y_train[i][0] = 1.
    y_train = y_train.reshape(x_train.shape[0], 1, 1, n_labels)

    test_var1 = pred.eval(feed_dict={x:x_train, y:y_train})
    # test_var2 = test_var.eval(feed_dict={x:x_train, y:y_train})

    # Logging every 100th iteration in the training process.
    # if i%1 == 0:
    #     test_var1 = dense1.eval(feed_dict={x:x_train, y:y_train})
        # test_var2 = end_points.eval(feed_dict={x:x_train, y:y_train})
        # print(test_var1.shape)
        #print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: x_train, y: y_train}) #, keep_prob: 0.5})


# a = x_input[0,0,:, :, 0]#[0]
# b = a[:, :][0]
# np.savetxt("/homes/mat10/Programming/OpenCV/frames/test.csv", a)
# a = test_var1[0][0][0]
# test_var1[:, 0, :, :, :]
