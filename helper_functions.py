import numpy as np
import cv2
import os

filedir = "/home/michael/Documents/MSc Machine Learning/ISO-Deep Lip Reading/Stafylakis_Tzimiropoulos/Tensorflow_Implementation/frames/mouths2"
subdirs = ['word1', 'word2']
files = "all"

# if files == 'all':
#     files = [[file for file in os.listdir(filedir+subdir) if file[-3:] == 'jpg'] for subdir in subdirs]

def get_data():
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

def make_path(*args):
    last_arg = args[-1]
    args = [arg + "/" for arg in args[:-1]]
    path = ''.join(args) + last_arg
    return path


# def get_data():
#     """
#     imports data and stacks them in the 3rd dimension
#     :return:
#     """
#     img = [cv2.imread(filedir + file, 0) for file in files]
#     img = np.array(img)
#     n, w, h = img.shape
#     return img.reshape(1, n, w, h, 1)

