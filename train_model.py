import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import pandas as pd
from model import front_end, concat_resnet_output, fully_connected_with_logits
from tfpipeline import get_batch

#--------------------------------------------------------------------------------------------------------------------#
# USER INPUT
# dataset_dir= "/homes/mat10/Desktop/tfrecords_test/ABOUT"
train_data_info = pd.read_csv("/data/mat10/ISO_Lipreading/data/LRW_TFRecords/train_data_info.csv")
options = {'is_training': True, 'batch_size': 32, 'num_classes': 500, 'num_epochs': 20,
           'crop_size': 112, 'horizontal_flip': True}
training = True
#--------------------------------------------------------------------------------------------------------------------#

number_of_steps = int(options['num_epochs'] * (train_data_info.shape[0] // options['batch_size']))
print("Total number of steps: %d" % number_of_steps)

paths = list(train_data_info['path'])

videos, labels = get_batch(paths, options)

prediction = front_end(videos)
prediction = concat_resnet_output(prediction)
prediction = fully_connected_with_logits(prediction, 500)

slim = tf.contrib.slim

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)
# d = sess.run([data])

tf.losses.softmax_cross_entropy(labels, prediction)
total_loss = slim.losses.get_total_loss()

# Create some summaries to visualize the training process:
# tf.scalar_summary('losses/Total Loss', total_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = slim.learning.create_train_op(total_loss,
                                         optimizer,
                                         summarize_gradients=True)
logging.set_verbosity(1)
slim.learning.train(train_op=train_op,
                    number_of_steps=number_of_steps,
                    logdir='ckpt/train',
                    save_summaries_secs=60,
                    save_interval_secs=600)

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)
# pred = sess.run([prediction])
