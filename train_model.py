import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import pandas as pd
from model import *
from tfpipeline import get_batch

#--------------------------------------------------------------------------------------------------------------------#
# USER INPUT
# TRAIN OPTIONS
# data_info_dir = "/homes/mat10/Desktop/tfrecords_test"
data_info_dir = "/data/mat10/ISO_Lipreading/data/LRW_TFRecords"
train_data_info = pd.read_csv(data_info_dir + "/train_data_info.csv").sample(frac=1)
# val_data_info = pd.read_csv(data_info_dir + "/val_data_info.csv").sample(frac=1)
# words_step1 = ['BRITISH', 'ATTACKS', 'HAVING', 'BIGGEST', 'REPORT', 'FORCES',
#        'WANTED', 'HOURS', 'CONCERNS', 'INFORMATION']
# train_data_info = train_data_info[train_data_info['word'].isin(words_step1)]
train_options = {'batch_size': 32, 'num_classes': 500, 'num_epochs': 20,
                 'crop_size': 112, 'horizontal_flip': True, 'shuffle': True}

# MODEL RESTORE OPTIONS
restore = True
# specify the model directory
modeldir = "/data/mat10/ISO_Lipreading/models/saved_models_full"
model = "model_full_epoch13"

# SAVE OPTIONS
savedir = "/data/mat10/ISO_Lipreading/models/saved_models_full"

# START AT EPOCH
start_epoch = 13
#--------------------------------------------------------------------------------------------------------------------#
print("Total number of train data: %d" % train_data_info.shape[0])
number_of_steps_per_epoch = train_data_info.shape[0] // train_options['batch_size']
number_of_steps = train_options['num_epochs'] * number_of_steps_per_epoch

train_paths = list(train_data_info['path'])

#--------------------------------------------------------------------------------------------------------------------#
# TRAINING
train_videos, train_labels = get_batch(train_paths, train_options)
prediction = frontend_3D(train_videos)
prediction = backend_resnet34(prediction)
prediction = blstm_2layer(prediction)
prediction = fully_connected_logits(prediction, train_options['num_classes'])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=prediction))
learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=0,
                                           decay_steps=number_of_steps_per_epoch//2,
                                           decay_rate=0.9, staircase=True)
train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#logfile = open('saved_models_full/train_history_log.csv', 'w')
#logfile.write('epoch, step, train_accuracy \n')

with tf.Session() as sess:

    # initialize the variables
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    # Add ops to save and restore all the variables.
    if restore:
        saver.restore(sess, modeldir + "/" + model)
        print("Model restored.")

    # print("saving model before training")
    # saver.save(sess=sess, save_path=savedir + "/model_full_epoch%d" % 0)
    # print("model saved")

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for epoch in range(start_epoch, start_epoch + train_options['num_epochs']):
        
        print("saving model for epoch %d - step %d" % (epoch, 0))
        saver.save(sess=sess, save_path=savedir + "/model_full_epoch%d" % epoch)
        print("model saved")
        
        for step in range(number_of_steps_per_epoch):
            _, loss = sess.run([train_step, cross_entropy])

            train_acc = sess.run(accuracy)
            print("epoch: %d of %d - step: %d of %d - loss: %.4f - train accuracy: %.4f"
                  % (epoch, train_options['num_epochs'], step, number_of_steps_per_epoch, loss, train_acc))


                
                #logfile.write('%d, %d, %.4f \n' % (epoch, step, train_acc))

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()

#logfile.close()

#--------------------------------------------------------------------------------------------------------------------#



# #------------------------------------------------------------------------------------#
# # COMMENTED OUT
# prediction = frontend_3D(videos)
# prediction = backend_resnet34(prediction)
# prediction = blstm_2layer(prediction)
# prediction = fully_connected_logits(prediction, options['num_classes'])
#
# slim = tf.contrib.slim
#
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # tf.train.start_queue_runners(sess=sess)
# # d = sess.run([data])
#
# tf.losses.softmax_cross_entropy(labels, prediction)
# total_loss = slim.losses.get_total_loss()
#
# # Create some summaries to visualize the training process:
# # tf.scalar_summary('losses/Total Loss', total_loss)
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
# train_op = slim.learning.create_train_op(total_loss,
#                                          optimizer,
#                                          summarize_gradients=True)
#
# logging.set_verbosity(1)
# slim.learning.train(train_op=train_op,
#                     number_of_steps=number_of_steps,
#                     logdir='ckpt/train',
#                     save_summaries_secs=60,
#                     save_interval_secs=600)
#------------------------------------------------------------------------------------#

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)
# pred = sess.run([prediction])
