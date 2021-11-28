import os
import sys
import cv2
import numpy as np
from helpers.labeled_jsons import load_labelme_image, plot_image, load_labelme_droplet_labels
import matplotlib.pyplot as plt
import glob
from detector.circle_detector import detect_circles


# JSON_PATH = os.path.abspath('../resources/105mm_60deg.6mxcodhz.000000')

# dsettings = {
#     'ds_coeff': 2,
#     'circle_mask_rad': 30,
#     'circle_mask_wdth': None,
#     'circle_mask_radoff_size': 5,
#     'pxcorr1': 90,
#     'pxcorr2': 90,
#     'peakfind_thr': 0.1,
#     'peakfind_min_max_nghbr': 30,
#     'debug': False
# }


# def preprocess_img(img: np.ndarray):
#     # img = cv2.normalize(img, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
#     clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(60, 60))
#     img = clahe.apply(img)
#     img[img < 50] = 0
#     return img


LM_PATHS = glob.glob(os.path.join(JSON_PATH, '*.json'))
for k, jpath in enumerate(JSON_PATH):
    img = load_labelme_image(jpath)
    pimg = preprocess_img(img)
    gt_droplets = load_labelme_droplet_labels(jpath)


    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(223)
    plt.hist(img, bins=256 ,histtype='step')

    plt.subplot(222)
    plt.imshow(pimg)
    plt.title('processed')
    plt.subplot(224)
    plt.hist(pimg, bins=256, histtype='step')
    plt.show()

    # todo - this thing will load the json with detector output and show images and labels



