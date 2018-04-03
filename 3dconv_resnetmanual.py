import numpy as np
import tensorflow as tf
from helper_functions import *
from resnet_model import *
# from tensorflow.contrib.slim.nets import resnet_v2


# USER INPUT - START ---------------------------------------------------#
width = 112
height = 112
depth = 1
num_frames = 28
n_labels = 512
batch_size = 2
datadir = ("/home/mat10/Documents/MSc Machine Learning/ISO-Deep Lip Reading/"
           "Stafylakis_Tzimiropoulos/Tensorflow_Implementation/frames/mouths2")
subdirs = ['word1', 'word2']
savedir = ("/home/mat10/Documents/MSc Machine Learning/ISO-Deep Lip Reading/"
           "Stafylakis_Tzimiropoulos/Tensorflow_Implementation/trained_models")
# USER INPUT - END ---------------------------------------------------- #

sess = tf.InteractiveSession()

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

# INPUT - OUTPUT
x = tf.placeholder(tf.float32, shape=[batch_size, num_frames, width, height, 1])
print("shape of input is %s" % x.get_shape)
y = tf.placeholder(tf.float32, shape=[batch_size, 1, 1, n_labels])
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

# RESNET
# Split the 3d convolution out to feed it to resnet
resnet_input_series = tf.unstack(h_pool1, axis=1)
print("number of elements in resnet_input_series is %s" % len(resnet_input_series))
print("shape of resnet_input_series element is %s" % resnet_input_series[0].get_shape())

# need to change that to 34 layer model
resnet_size = 34
num_frames = len(resnet_input_series)
scopes = ["res_frame%d"%i for i in range(num_frames)]
res_out = []
for i, resnet_input in enumerate(resnet_input_series):
    with tf.variable_scope(scopes[i]):
        resnet = Model(resnet_size=resnet_size, bottleneck=False, num_classes=512, num_filters=64,
                       kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2,
                       second_pool_size=7, second_pool_stride=1, block_sizes=get_block_sizes(resnet_size),
                       block_strides=[1, 2, 2, 2], final_size=512)
        features = resnet.__call__(resnet_input, True)
        features = tf.reshape(features, [batch_size, 1, 1, 512])
        res_out.append(features)

print("shape after resnet is %s" % features.get_shape())

# add resnet outputs per frame. This vector estimates resnets prediction on the word spoken
result = tf.add_n(res_out)

predictions = tf.nn.softmax(result)
print("shape of predictions is %s" % predictions.get_shape())

# TRAINING AND EVALUATION
# set up for optimization (optimizer:ADAM)
test_var = predictions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=test_var))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(test_var, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for epoch in range(5):
    print(epoch)

    # GET BATCH DATA
    x_train = get_data(datadir, subdirs)
    y_train = np.array(x_train.shape[0] * [np.zeros(n_labels).astype(np.float32)])
    for i in range(x_train.shape[0]):
        y_train[i][0] = 1.
    y_train = y_train.reshape(x_train.shape[0], 1, 1, n_labels)

    # RUN SINGLE TRAIN STEP
    train_step.run(feed_dict={x: x_train, y: y_train})  # , keep_prob: 0.5})

    test_var1 = predictions.eval(feed_dict={x:x_train})#, y:y_train})
    # test_var2 = test_var.eval(feed_dict={x:x_train, y:y_train})

    # Logging every 100th iteration in the training process.
    # if i%1 == 0:
    #     test_var1 = dense1.eval(feed_dict={x:x_train, y:y_train})
        # test_var2 = end_points.eval(feed_dict={x:x_train, y:y_train})
        # print(test_var1.shape)
        #print("step %d, training accuracy %g"%(i, train_accuracy))


# saver.save(sess, savedir + "/model01")  #filename ends with .ckpt
# print("Model saved in path: %s" % savedir)

# a = x_input[0,0,:, :, 0]#[0]
# b = a[:, :][0]
# np.savetxt("/homes/mat10/Programming/OpenCV/frames/test.csv", a)
# a = test_var1[0][0][0]
# test_var1[:, 0, :, :, :]
