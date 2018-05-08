from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pandas as pd
import tensorflow as tf
# from pathlib import Path
# from inception_processing import distort_color


def image_left_right_flip(image):
    return tf.image.flip_left_right(image)
    # images_list = tf.unstack(video)
    # for i in range(len(images_list)):
    #     images_list[i] = tf.image.flip_left_right(images_list[i])
    # return tf.stack(images_list)


def video_left_right_flip(video):
    return tf.map_fn(image_left_right_flip, video)


def normalize(videos):
    # return videos * (1. / 255.) - 0.5
    return (videos - 127.5) / 50


def get_batch(paths, options):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    shuffle = options['shuffle']
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    crop_size = options['crop_size']
    horizontal_flip = options['horizontal_flip']

    # root_path = Path(dataset_dir) / split_name
    # paths = [str(x) for x in root_path.glob('*.tfrecords')]

    filename_queue = tf.train.string_input_producer(paths, shuffle=shuffle)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'video': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    video = tf.cast(tf.decode_raw(features['video'], tf.uint8), tf.float32) #/ 255.
    label = features['label']#tf.decode_raw(features['label'], tf.int64)

    # Number of threads should always be one, in order to load samples
    # sequentially.
    videos, labels = tf.train.batch(
        [video, label], batch_size, num_threads=1, capacity=1000, dynamic_pad=True)

    videos = tf.reshape(videos, (batch_size, 29, 118, 118, 1))
    #labels = tf.reshape(labels, (batch_size,  1))
    labels = tf.contrib.layers.one_hot_encoding(labels, num_classes)

    # if is_training:
        # resized_image = tf.image.resize_images(frame, [crop_size, 110])
        # random cropping
    if crop_size is not None:
        videos = tf.random_crop(videos, [batch_size, 29, crop_size, crop_size, 1])
    # random left right flip
    if horizontal_flip:
        sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        option = tf.less(sample, 0.5)
        videos = tf.cond(option,
                         lambda: tf.map_fn(video_left_right_flip, videos),
                         lambda: tf.map_fn(tf.identity, videos))
            # lambda: video_left_right_flip(videos),
            # lambda: tf.identity(videos))
    videos = normalize(videos) #tf.cast(videos, tf.float32) * (1. / 255.) - 0.5

    return videos, labels


# # dataset_dir= "/homes/mat10/Desktop/tfrecords_test/ABOUT"
# train_data_info = pd.read_csv("/homes/mat10/Desktop/tfrecords_test/train_data_info.csv")
# options = {'is_training': True, 'batch_size': 10, 'num_classes': 500,
#            'crop_size': 112, 'horizontal_flip': True}
#
# paths = list(train_data_info['path'])
# videos, labels = get_batch(paths, options)
#
# slim = tf.contrib.slim
#
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # tf.train.start_queue_runners(sess=sess)
# # d = sess.run([data])
#
# prediction = get_model(videos)
#
# tf.losses.softmax_cross_entropy(labels, prediction)
# total_loss = slim.losses.get_total_loss()
# optimizer = tf.train.AdamOptimizer(0.001)
# train_op = slim.learning.create_train_op(total_loss,
#                                          optimizer,
#                                          summarize_gradients=True)
#
#
# slim.learning.train(train_op,
#                     'ckpt/train',
#                     save_summaries_secs=60,
#                     save_interval_secs=300)
#                     #number_of_steps=max_steps)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)
# pred = sess.run([prediction])
