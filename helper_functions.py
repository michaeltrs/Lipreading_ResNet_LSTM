import numpy as np
import cv2
import os
from scipy.io import loadmat
from shutil import copyfile

# filedir = ("/homes/mat10/Documents/MSc_Machine_Learning/"
#            "ISO_Deep_Lip_Reading/Stafylakis_Tzimiropoulos/"
#            "Tensorflow_Implementation/frames/mouths2")
# subdirs = ['word1', 'word2']
# files = "all"

# if files == 'all':
#     files = [[file for file in os.listdir(filedir+subdir) if file[-3:] == 'jpg'] for subdir in subdirs]
def get_npy_data(paths):
    """
    imports data and stacks them in the 3rd dimension
    :return:
    """
    data = np.array([np.load(path) for path in paths])
    data = data[:, :, 3:115, 3:115]
    nbatch, nframes, width, height = data.shape
    data = data.reshape(nbatch, nframes, width, height, 1)
    return data


def get_data(filedir, subdirs, files='all'):
    """
    imports data and stacks them in the 3rd dimension
    :return:
    """
    img = [[cv2.imread(make_path(filedir, subdir, file), 0)
            for file in os.listdir(make_path(filedir, subdir)) if file[-3:] == 'jpg']
            for subdir in subdirs]
    img = np.array(img)
    nbatches, n, w, h = img.shape
    return img.reshape(nbatches, n, w, h, 1)

# old function
# def get_data(filedir, subdirs, files='all'):
#     """
#     imports data and stacks them in the 3rd dimension
#     :return:
#     """
#     img = [[cv2.imread(make_path(filedir, subdir, file), 0)
#             for file in os.listdir(make_path(filedir, subdir)) if file[-3:] == 'jpg']
#             for subdir in subdirs]
#     img = np.array(img)
#     nbatches, n, w, h = img.shape
#     return img.reshape(nbatches, n, w, h, 1)

def make_path(*args):
    last_arg = args[-1]
    args = [arg + "/" for arg in args[:-1]]
    path = ''.join(args) + last_arg
    return path


def video2frames(filedir, file, savedir):
    vidcap = cv2.VideoCapture(filedir + file)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(savedir + "%s_frame%d.jpg" % (file[:-5], count), image)     # save frame as JPEG file  count += 1
        count += 1
    print('Saved %d frames' %count)


def extract_mouth_roi(files_dir, dataset, word, file, lms_dir, resolution=118,
                      save_file_name=None):
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
    if save_file_name:
        np.save(save_file_name, mouth_roi)
    else:
        return mouth_roi


# def save_data_from_videos(data_root_dir, lms_dir, save_root_dir, words='all', datasets='all'):
#     if words == "all":
#         words = os.listdir(data_root_dir)  # list of words
#     if datasets == "all":
#         datasets = ['train', 'val', 'test']
#     for dataset in datasets: # train, test, val
#         for word in words: # ABOUT, ...
#             print(dataset + "-" + word)
#             # check if directory already exists
#             if not os.path.isdir(save_root_dir + "/" + word):
#                 os.makedirs(save_root_dir + "/" + word)
#             if not os.path.isdir(save_root_dir + "/" + word + "/" + dataset):
#                 os.makedirs(save_root_dir + "/" + word + "/" + dataset)
#             files_dir = data_root_dir + "/" + word + "/" + dataset + "/"
#             files_mp4 = [file for file in os.listdir(files_dir) if file[-3:] == 'mp4']
#             files_txt = [file for file in os.listdir(files_dir) if file[-3:] == 'txt']
#             # file = files_mp4[0]
#             for i, file in enumerate(files_mp4):
#                 save_dir = save_root_dir + "/" + word + "/" + dataset + "/"
#                 # check if save directory already exists
#                 # if not os.path.isdir(save_dir):
#                 #     os.makedirs(save_dir)
#                 # copy txt file to save directory
#                 file_txt = files_txt[i]
#                 copyfile(files_dir + file_txt, save_dir + file_txt)
#                 # split mp4 to frames and save to save directory
#                 # video2frames(files_dir, file, save_dir)
#                 save_file_name = save_root_dir + "/" + word + "/" + dataset + "/" + file[:-4] + ".npy"
#                 extract_mouth_roi(files_dir, dataset, word, file, lms_dir,
#                                   resolution=118, save_file_name=save_file_name)

# def get_data():
#     """
#     imports data and stacks them in the 3rd dimension
#     :return:
#     """
#     img = [cv2.imread(filedir + file, 0) for file in files]
#     img = np.array(img)
#     n, w, h = img.shape
#     return img.reshape(1, n, w, h, 1)

