import numpy as np
import tensorflow as tf
import pandas as pd
from model import * #frontend_3D, backend_resnet34, concat_resnet_output, fully_connected_logits
from tfpipeline import get_batch

#--------------------------------------------------------------------------------------------------------------------#
# USER INPUT
# dataset_dir= "/homes/mat10/Desktop/tfrecords_test/ABOUT"
# data_info_dir = "/homes/mat10/Desktop/tfrecords_test"
data_info_dir = "/data/mat10/ISO_Lipreading/data/LRW_TFRecords"
val_data_info = pd.read_csv(data_info_dir + "/val_data_info.csv").sample(frac=1)
train_data_info = pd.read_csv(data_info_dir + "/train_data_info.csv").sample(frac=1)
options = {'is_training': False, 'batch_size': 100, 'num_classes': 500, 'num_epochs': 1,
           'crop_size': 112, 'horizontal_flip': False, "shuffle": False}
# specify the model directory
# srcdir = "/data/mat10/ISO_Lipreading/models/saved_models_full"
model_dir = "/data/mat10/ISO_Lipreading/models/saved_models_full/"
model_ = model_dir + "model_full_epoch"
savedir = "/data/mat10/ISO_Lipreading/models/evaluation"
#--------------------------------------------------------------------------------------------------------------------#

# print("Total number of train data: %d" % data_info.shape[0])
number_of_steps_per_epoch = (val_data_info.shape[0] // options['batch_size'])
# print("Total number of steps: %d" % number_of_steps)

# modelid = 17
# data_info = val_data_info
for modelid in reversed(range(1, 17)):

    print("Evaluating model %d" % modelid)
    model = model_ + str(modelid)

    for dataid, data_info in enumerate([train_data_info]):#, train_data_info]):
        dataid = 1
        paths = list(data_info['path'])[:25000]

        videos, labels = get_batch(paths, options)

        prediction = frontend_3D(videos)
        prediction = backend_resnet34(prediction)
        prediction = blstm_2layer(prediction)
        prediction = fully_connected_logits(prediction, options['num_classes'])


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction))
        true_class = tf.argmax(labels, axis=1)
        predicted_class = tf.argmax(prediction, axis=1)
        correct_prediction = tf.equal(true_class, predicted_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()

        sess.run(tf.initialize_all_variables())

        tf.train.start_queue_runners(sess=sess)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        saver.restore(sess, model)
        print("Model restored.")

        metrics_ = []
        labels_ = []
        predicted_ = []
        for i in range(number_of_steps_per_epoch):
            print("model %d - dataid %d - step %d of %d" % (modelid, dataid, i, number_of_steps_per_epoch))
            loss, acc = sess.run([cross_entropy, accuracy])
            #lab, pred, loss, acc = sess.run([true_class, predicted_class, cross_entropy, accuracy])
            metrics_.append([loss, acc])
            #labels_.append(lab)
            #predicted_.append(pred)

        metrics_ = np.array(metrics_)
        np.save(savedir + "/loss_accuracy_model%d_data%d.npy" % (modelid, dataid), metrics_)

        #labels_ = np.array(labels_)
        #np.save(savedir + "/truelabels_model%d_data%d.npy" % (modelid, dataid), labels_)

        #predicted_ = np.array(predicted_)
        #np.save(savedir + "/predictedlabels_model%d_data%d.npy" % (modelid, dataid), predicted_)

        tf.reset_default_graph()


