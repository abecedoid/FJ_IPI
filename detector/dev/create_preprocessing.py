import os
import argparse
import sys
import cv2
import numpy as np

sys.path.append('../..')
sys.path.append('..')
from helpers.labeled_jsons import load_labelme_image, plot_image
import matplotlib.pyplot as plt


JSON_PATH = os.path.abspath('../../resources/105mm_60deg.6mxcodhz.000000')


def preprocess_img(img: np.ndarray):
    # img = cv2.normalize(img, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(60, 60))
    img = clahe.apply(img)
    img[img < 50] = 0
    return img


try:
    path2json = os.path.abspath(JSON_PATH)
except Exception as e:
    print('Invalid path, {}'.format(e))
    sys.exit()

if os.path.isdir(path2json):
    json_filenames = os.listdir(path2json)
    for filename in json_filenames:
        path = os.path.join(path2json, filename)
        print('Loading {} ...'.format(path2json))
        img = load_labelme_image(path)
        pimg = preprocess_img(img)

        plt.figure()
        plt.subplot(221)
        plt.imshow(img)
        plt.title('original')
        plt.subplot(223)
        plt.hist(img, bins=256,histtype='step')

        plt.subplot(222)
        plt.imshow(pimg)
        plt.title('processed')
        plt.subplot(224)
        plt.hist(pimg, bins=256, histtype='step')
        plt.show()
