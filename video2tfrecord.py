# from helper_functions import extract_mouth_roi
from multiprocessing import Pool
from shutil import copyfile
import os
import logging
from scipy.io import loadmat
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(video, label, save_file_name):

    """
    convert videos and labels to tf serialized data and save to
    TFRecord binary file
    # Args:
    # videos        List of 29 frames np.ndarray
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    """
    # print("Converting: " + out_path)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_file_name) as writer:

        # Convert the image to raw bytes.
        video_bytes = video.tostring()

        # Create a dict with the data we want to save in the
        # TFRecords file. You can add more relevant data here.
        data = \
            {
                'video': wrap_bytes(video_bytes),
                'label': wrap_int64(label)
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)

        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)

        # Serialize the data.
        serialized = example.SerializeToString()

        # Write the serialized data to the TFRecords file.
        writer.write(serialized)


def extract_mouth_roi(files_dir, dataset, word, file, lms_dir, save_file_name, resolution=118):
    # facial Landmarks
    lms_file = 'lipread_mp4__%s__%s__%s.mat' % (word, dataset, file[:-4])
    lms = loadmat(lms_dir + "/" + lms_file)
    lms = lms['pts']
    # if lms.shape[0] != 136: # LOG if lms.shape[0] != 136
    #     print()
    lms_x = lms[:68, :]
    lms_y = lms[68:, :]
    mouth_center_x = int(np.median(np.mean(lms_x[48:68, :], axis=1)))
    mouth_center_y = int(np.median(np.mean(lms_y[48:68, :], axis=1)))
    # video
    vidcap = cv2.VideoCapture(files_dir + file)
    success, image = vidcap.read()
    gray_image = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
    count = 1
    while success:
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            gray_image.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            count += 1
        # cv2.imwrite(savedir + "%s_frame%d.jpg" % (file[:-5], count), image)     # save frame as JPEG file  count += 1
    if count != 29:
        print("File %s has %d frames" % (files_dir + file, count))

    gray_image = np.array(gray_image)

    # mouth roi
    # dimensions of square mouth region, 112 +- 6 pixels for data augmentation later
    dim = resolution // 2

    mouth_roi = gray_image[:, mouth_center_y - dim : mouth_center_y + dim,
                           mouth_center_x - dim : mouth_center_x + dim]
    # either save on disk or return array
    label = data_info[data_info['word'] == word]['class'].values[0]

    convert(mouth_roi, label, save_file_name)


def save_data_from_videos(words='all'):
    if words == "all":
        words = os.listdir(data_root_dir)  # list of words
    for dataset in datasets: # train, test, val
        for word in words: # ABOUT, ...
            #try:
            print(dataset + "-" + word)
            files_dir = data_root_dir + "/" + word + "/" + dataset + "/"
            if os.path.isdir(files_dir):
                save_dir = save_root_dir + "/" + word + "/" + dataset + "/"
                # check if directory already exists
                if not os.path.isdir(save_root_dir + "/" + word):
                    os.makedirs(save_root_dir + "/" + word)
                if not os.path.isdir(save_root_dir + "/" + word + "/" + dataset):
                    os.makedirs(save_root_dir + "/" + word + "/" + dataset)

                files_mp4 = [file for file in os.listdir(files_dir) if file[-3:] == 'mp4']
                files_txt = [file for file in os.listdir(files_dir) if file[-3:] == 'txt']
                for file_txt in files_txt:
                    # copy txt file to save directory
                    copyfile(files_dir + file_txt, save_dir + file_txt)
                # file = files_mp4[0]
                for i, file in enumerate(files_mp4):
                    # split mp4 to frames and save to save directory
                    # video2frames(files_dir, file, save_dir)
                    save_file_name = save_root_dir + "/" + word + "/" + dataset + "/" + file[:-4] + ".tfrecords"
                    extract_mouth_roi(files_dir, dataset, word, file, lms_dir,
                                      resolution=118, save_file_name=save_file_name)
                # except:
                #     logging.exception('Got exception on main handler')
            else:
                logging.info("%s has no %s set" % (word, dataset))



def split_list(list_, num_chunks):
    chunk_size = len(list_) // num_chunks
    # remainder = len(list_) % num_chunks
    res = []
    for i in range(num_chunks):
        if i == num_chunks-1:
            res.append(list_[i*chunk_size:])
        else:
            res.append(list_[i*chunk_size:(i+1)*chunk_size])
    return res


if __name__ == '__main__':

    # Facial landmarks directory
    lms_dir = "/vol/atlas/homes/thanos/bbc/landmarks/2017_9_lip_reading/lip_reading_pts"
    # Root directory of data
    data_root_dir = "/vol/atlas/homes/thanos/bbc/lipread_mp4"
    save_root_dir = "/data/mat10/ISO_Lipreading/data/LRW_TFRecords"
    #"/homes/mat10/Desktop/tfrecords_test"
    data_info_path = ("/data/mat10/ISO_Lipreading/data/LRW_TFRecords/data_info.csv")
    data_info = pd.read_csv(data_info_path)
    datasets = list(data_info["dataset"].unique())

    LOG_FILENAME = save_root_dir + '/errors.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    words = os.listdir(data_root_dir)
    # datasets = ['train', 'val', 'test']

    num_cores = 8
    words_split = split_list(words, num_cores)

    logging.info('Start')

    pool = Pool(num_cores)
    pool.map(save_data_from_videos, words_split)
























# -------------------------------------------------------------------------- #
# from helper_functions import extract_mouth_roi
# from multiprocessing import Pool
# from shutil import copyfile
# import os
# import logging
#
#
# def save_data_from_videos(words='all'):
#     if words == "all":
#         words = os.listdir(data_root_dir)  # list of words
#     for dataset in datasets: # train, test, val
#         for word in words: # ABOUT, ...
#             try:
#                 print(dataset + "-" + word)
#                 # check if directory already exists
#                 if not os.path.isdir(save_root_dir + "/" + word):
#                     os.makedirs(save_root_dir + "/" + word)
#                 if not os.path.isdir(save_root_dir + "/" + word + "/" + dataset):
#                     os.makedirs(save_root_dir + "/" + word + "/" + dataset)
#                 files_dir = data_root_dir + "/" + word + "/" + dataset + "/"
#                 files_mp4 = [file for file in os.listdir(files_dir) if file[-3:] == 'mp4']
#                 files_txt = [file for file in os.listdir(files_dir) if file[-3:] == 'txt']
#                 # file = files_mp4[0]
#                 for i, file in enumerate(files_mp4):
#                     save_dir = save_root_dir + "/" + word + "/" + dataset + "/"
#                     # copy txt file to save directory
#                     file_txt = files_txt[i]
#                     copyfile(files_dir + file_txt, save_dir + file_txt)
#                     # split mp4 to frames and save to save directory
#                     # video2frames(files_dir, file, save_dir)
#                     save_file_name = save_root_dir + "/" + word + "/" + dataset + "/" + file[:-4] + ".npy"
#                     extract_mouth_roi(files_dir, dataset, word, file, lms_dir,
#                                       resolution=118, save_file_name=save_file_name)
#             except:
#                 logging.exception('Got exception on main handler')
#
#
# def split_list(list_, num_chunks):
#     chunk_size = len(list_) // num_chunks
#     # remainder = len(list_) % num_chunks
#     res = []
#     for i in range(num_chunks):
#         if i == num_chunks-1:
#             res.append(list_[i*chunk_size:])
#         else:
#             res.append(list_[i*chunk_size:(i+1)*chunk_size])
#     return res
#
#
# if __name__ == '__main__':
#
#     # Facial landmarks directory
#     lms_dir = "/vol/atlas/homes/thanos/bbc/landmarks/2017_9_lip_reading/lip_reading_pts"
#     # Root directory of data
#     data_root_dir = "/vol/atlas/homes/thanos/bbc/lipread_mp4"
#     save_root_dir = "/data/mat10/ISO_Lipreading/data/LRW"
#     LOG_FILENAME = save_root_dir + '/errors.log'
#     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
#
#     words = os.listdir(data_root_dir)
#     datasets = ['train', 'val', 'test']
#
#     num_cores = 8
#     words_split = split_list(words, num_cores)
#
#     logging.info('Start')
#
#     pool = Pool(num_cores)
#     pool.map(save_data_from_videos, words_split)

