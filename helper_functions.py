import numpy as np
import cv2
import os

filedir = "/homes/mat10/Programming/OpenCV/frames/mouths2/"
files = "all"

if files == 'all':
    files = [file for file in os.listdir(filedir) if file[-3:] == 'jpg']


def get_data():
    """
    imports data and stacks them in the 3rd dimension
    :return:
    """
    img = [cv2.imread(filedir + file, 0) for file in files]
    img = np.array(img)
    n, w, h = img.shape
    return img.reshape(1, n, w, h, 1)

