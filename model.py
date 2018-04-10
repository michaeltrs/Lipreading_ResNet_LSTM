import tensorflow as tf
# from helper_functions import *
from resnet_model import ResNet, get_block_sizes
# from tensorflow.contrib.slim.nets import resnet_v2


def front_end(x_input, training=True):

    BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()
    NUM_CLASSES = 500

    # 3D CONVOLUTION
    # First Convolution Layer - Image input, use 64 3D filters of size 5x5x5
    # shape of weights is (dx, dy, dz, #in_filters, #out_filters)
    W_conv1 = tf.get_variable("W_conv1", initializer=tf.truncated_normal(shape=[5, 7, 7, 1, 64], stddev=0.1))
    b_conv1 = tf.get_variable("b_conv1", initializer=tf.constant(0.1, shape=[64]))
    # apply first convolution
    z_conv1 = tf.nn.conv3d(x_input, W_conv1, strides=[1, 1, 2, 2, 1], padding='SAME') + b_conv1
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
    batch_size, seq_length, height, width, channels = h_pool1.get_shape().as_list()

    video_input = tf.reshape(h_pool1, (batch_size * seq_length, height, width, channels))

    #  = tf.cast(video_input, tf.float32)
    resnet_size = 34
    resnet = ResNet(resnet_size=resnet_size, bottleneck=False, num_classes=512, num_filters=64,
                    kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2,
                    second_pool_size=7, second_pool_stride=1, block_sizes=get_block_sizes(resnet_size),
                    block_strides=[1, 2, 2, 2], final_size=512)
    features = resnet.__call__(video_input, True)
    # features, end_points = resnet_v2.resnet_v1_50(video_input, None)
    features = tf.reshape(features, (BATCH_SIZE, NUM_FRAMES, int(features.get_shape()[1])))

    print("shape after resnet is %s" % features.get_shape())

    return features


def concat_resnet_output(resnet_out):
    # add resnet outputs per frame. This vector estimates resnets prediction on the word spoken
    return tf.reduce_sum(resnet_out, axis=1)


def fully_connected_with_logits(x_input, out_size):
    # fully connected layer (512 -> 500)
    dense = tf.layers.dense(inputs=x_input, units=out_size, activation=tf.nn.relu)
    predictions = tf.nn.softmax(dense)
    print("shape of predictions is %s" % predictions.get_shape())
    return predictions


# def get_model(x, training=True):
#
#
#     # 3D CONVOLUTION
#     # First Convolution Layer - Image input, use 64 3D filters of size 5x5x5
#     # shape of weights is (dx, dy, dz, #in_filters, #out_filters)
#     W_conv1 = tf.get_variable("W_conv1", initializer=tf.truncated_normal(shape=[5, 7, 7, 1, 64], stddev=0.1))
#     b_conv1 = tf.get_variable("b_conv1", initializer=tf.constant(0.1, shape=[64]))
#     # apply first convolution
#     z_conv1 = tf.nn.conv3d(x, W_conv1, strides=[1, 1, 2, 2, 1], padding='SAME') + b_conv1
#     # apply batch normalization
#     z_conv1_bn = tf.contrib.layers.batch_norm(z_conv1,
#                     data_format='NHWC',  # Matching the "cnn" tensor shape
#                     center=True,
#                     scale=True,
#                     is_training=training,
#                     scope='cnn3d-batch_norm')
#     # apply relu activation
#     h_conv1 = tf.nn.relu(z_conv1_bn)
#     print("shape after 1st convolution is %s" % h_conv1.get_shape)
#
#     # apply max pooling
#     h_pool1 = tf.nn.max_pool3d(h_conv1,
#                                strides=[1, 1, 2, 2, 1],
#                                ksize=[1, 3, 3, 3, 1],
#                                padding='SAME')
#     print(h_pool1.get_shape)
#     print("shape after 1st pooling is %s" % h_pool1.get_shape)
#
#
#     # Split the 3d convolution out to feed it to resnet
#     resnet_input_series = tf.unstack(h_pool1, axis=1)
#     print("number of elements in resnet_input_series is %s" % len(resnet_input_series))
#     print("shape of resnet_input_series element is %s" % resnet_input_series[0].get_shape())
#
#
#     # RESNET
#     # need to change that to 34 layer model
#     num_frames = len(resnet_input_series)
#     scopes = ["res_frame%d"%i for i in range(num_frames)]
#     # res_out = []
#     dense_out = [] # performs dimensionality reduction to the output of resnet (512 -> 256)
#     for i, resnet_input in enumerate(resnet_input_series[:2]):    # REMOVE [:2]
#         with tf.variable_scope(scopes[i]):
#             features, end_points = resnet_v2.resnet_v2_50(resnet_input, num_classes=512)
#             # res_out.append(features)
#             dense_out.append(tf.layers.dense(inputs=features, units=256, activation=tf.nn.relu))
#
#     print("shape after resnet is %s" % features.get_shape())
#
#     # add resnet outputs per frame. This vector estimates resnets prediction on the word spoken
#     # result = tf.add_n(res_out)
#     # predictions = tf.nn.softmax(result)
#     # print("shape of predictions is %s" % predictions.get_shape())
#
#     # 2-layer BiLSTM
#     dense_out = tf.concat(dense_out, axis=1)
#     # Define input for forward and backward LSTM
#     dense_out_forw = tf.squeeze(dense_out, axis=2)
#     dense_out_back = tf.reverse(dense_out_forw, axis=[1])
#     # create 2 layer LSTMCells
#     rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
#
#     # create a RNN cell composed sequentially of a number of RNNCells
#     multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
#     multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
#
#     # 'outputs' is a tensor of shape [batch_size, max_time, 256]
#     # 'state' is a N-tuple where N is the number of LSTMCells containing a
#     # tf.contrib.rnn.LSTMStateTuple for each cell
#     outputs_forw, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
#                                        inputs=dense_out_forw,
#                                        dtype=tf.float32)
#     outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
#                                        inputs=dense_out_back,
#                                        dtype=tf.float32)
#
#     # get only the last output from lstm
#     # lstm_out = tf.transpose(lstm_out, [1, 0, 2])
#     last_forw = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1])-1, axis=1)
#     last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1])-1, axis=1)
#
#     bilstm_out = tf.concat([last_forw, last_back], axis=1)
#     print("shape of bilstm output is %s" % bilstm_out.get_shape())
#
#     linear_out = tf.layers.dense(inputs=bilstm_out, units=500, activation=tf.nn.relu)
#
#     pred = tf.nn.softmax(linear_out)
#
#     return pred