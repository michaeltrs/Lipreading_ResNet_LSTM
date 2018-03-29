import numpy as np
import tensorflow as tf
from helper_functions import *
from tensorflow.contrib.slim.nets import resnet_v2

# USER INPUT - START ---------------------------------------------------#
width = 112
height = 112
depth = 1
num_frames = 28
n_labels = 500
savedir = ("/data/mat10/ISO_Lipreading/models/saved_models")
    # ("/homes/mat10/Documents/MSc_Machine_Learning/"
    #        "ISO-Deep Lip Reading/Stafylakis_Tzimiropoulos/"
    #        "Tensorflow_Implementation/trained_models")
filedir = ("/homes/mat10/Documents/MSc_Machine_Learning/"
           "ISO_Deep_Lip_Reading/Stafylakis_Tzimiropoulos/"
           "Tensorflow_Implementation/frames/mouths2")
subdirs = ['word1', 'word2']
# USER INPUT - END ---------------------------------------------------- #

sess = tf.InteractiveSession()

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

# INPUT - OUTPUT
x = tf.placeholder(tf.float32, shape=[None, num_frames, width, height, 1])
print("shape of input is %s" % x.get_shape)
y = tf.placeholder(tf.float32, shape=[None, 1, 1, n_labels])
print("shape of output is %s" % y.get_shape)

# 3D CONVOLUTION
# First Convolution Layer - Image input, use 64 3D filters of size 5x5x5
# shape of weights is (dx, dy, dz, #in_filters, #out_filters)
W_conv1 = tf.get_variable("W_conv1", initializer=tf.truncated_normal(shape=[5, 7, 7, 1, 64], stddev=0.1))
b_conv1 = tf.get_variable("b_conv1", initializer=tf.constant(0.1, shape=[64]))
# apply first convolution
z_conv1 = tf.nn.conv3d(x, W_conv1, strides=[1, 1, 2, 2, 1], padding='SAME') + b_conv1
# apply batch normalization
z_conv1_bn = tf.contrib.layers.batch_norm(z_conv1,
                data_format='NHWC',  # Matching the "cnn" tensor shape
                center=True,
                scale=True,
                is_training=training,
                scope='cnn3d-batch_norm')
# apply relu activation
h_conv1 = tf.nn.relu(z_conv1_bn)
print("shape after 1st convolution is %s" % h_conv1.get_shape)

# apply max pooling
h_pool1 = tf.nn.max_pool3d(h_conv1,
                           strides=[1, 1, 2, 2, 1],
                           ksize=[1, 3, 3, 3, 1],
                           padding='SAME')
print(h_pool1.get_shape)
print("shape after 1st pooling is %s" % h_pool1.get_shape)


# Split the 3d convolution out to feed it to resnet
resnet_input_series = tf.unstack(h_pool1, axis=1)
print("number of elements in resnet_input_series is %s" % len(resnet_input_series))
print("shape of resnet_input_series element is %s" % resnet_input_series[0].get_shape())


# RESNET
# need to change that to 34 layer model
num_frames = len(resnet_input_series)
scopes = ["res_frame%d"%i for i in range(num_frames)]
# res_out = []
dense_out = [] # performs dimensionality reduction to the output of resnet (512 -> 256)
for i, resnet_input in enumerate(resnet_input_series[:2]):    # REMOVE [:2]
    with tf.variable_scope(scopes[i]):
        features, end_points = resnet_v2.resnet_v2_50(resnet_input, num_classes=512)
        # res_out.append(features)
        dense_out.append(tf.layers.dense(inputs=features, units=256, activation=tf.nn.relu))

print("shape after resnet is %s" % features.get_shape())

# add resnet outputs per frame. This vector estimates resnets prediction on the word spoken
# result = tf.add_n(res_out)
# predictions = tf.nn.softmax(result)
# print("shape of predictions is %s" % predictions.get_shape())

# 2-layer BiLSTM
dense_out = tf.concat(dense_out, axis=1)
# Define input for forward and backward LSTM
dense_out_forw = tf.squeeze(dense_out, axis=2)
dense_out_back = tf.reverse(dense_out_forw, axis=[1])
# create 2 layer LSTMCells
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs_forw, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                   inputs=dense_out_forw,
                                   dtype=tf.float32)
outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
                                   inputs=dense_out_back,
                                   dtype=tf.float32)

# get only the last output from lstm
# lstm_out = tf.transpose(lstm_out, [1, 0, 2])
last_forw = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1])-1, axis=1)
last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1])-1, axis=1)

bilstm_out = tf.concat([last_forw, last_back], axis=1)
print("shape of bilstm output is %s" % bilstm_out.get_shape())

linear_out = tf.layers.dense(inputs=bilstm_out, units=500, activation=tf.nn.relu)

pred = tf.nn.softmax(linear_out)
pred = tf.reshape(pred, tf.shape(y))


# Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
test_var = pred #dense1
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=test_var))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(test_var, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for epoch in range(5):
    print(epoch)


    x_train = get_data(filedir, subdirs)
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

# saver.save(sess, savedir + "/model01")  #filename ends with .ckpt
# print("Model saved in path: %s" % savedir)
#
# a = x_input[0,0,:, :, 0]#[0]
# b = a[:, :][0]
# np.savetxt("/homes/mat10/Programming/OpenCV/frames/test.csv", a)
# a = test_var1[0][0][0]
# test_var1[:, 0, :, :, :]
