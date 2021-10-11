import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import scipy.ndimage as nd
import skimage.feature as ftr


def detect_circles(im, c_diam=130, bckg_thr=20, alpha=1.3, beta=40, corr_thr=0.001):
    # DOWNSAMPLE
    if np.product(im.shape) > np.product((512, 512)):
        old_res = im.shape
        ds_ratio = np.max(im.shape) / 512
        if ds_ratio < 1.0:
            raise Exception('Failed the downsample procedure, need to program it better')
        new_res = (int(im.shape[0] / ds_ratio), int(im.shape[1] / ds_ratio))
        imd = cv2.resize(im, new_res)
    else:
        ds_ratio = 1.0
        imd = im

    # CLEAR BACKGROUND & ENHANCE
    imd[imd < bckg_thr] = 0
    imd = cv2.equalizeHist(imd)
    imd = cv2.convertScaleAbs(imd, alpha=alpha, beta=beta)

    # cv2.imshow('edge image', imd.astype('uint8'))
    # cv2.waitKey()

    # MORPHOLOGICAL OPS -> CREATING A "CIRCLE" IMAGE - DILATION & EROSION
    gsel = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 255., 255., 0., 0., 0., 0.],
                     [0., 0., 0., 255., 255., 255., 255., 0., 0., 0.],
                     [0., 0., 0., 255., 255., 255., 255., 0., 0., 0.],
                     [0., 0., 0., 0., 255., 255., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    dimg = nd.grey_dilation(imd.astype(float), gsel.shape, structure=gsel.astype(float))
    edimg = nd.grey_erosion(dimg, gsel.shape, structure=gsel.astype(float))

    # cv2.imshow('edge image', edimg.astype('uint8'))
    # cv2.waitKey()

    # CORRELATION
    d = int(c_diam / ds_ratio)  # diameter of downsampled interfero-circle
    mask_res = (2 * d, 2 * d)
    mask_cntr = (d, d)
    mask = np.zeros(mask_res)
    mask = cv2.circle(mask, mask_cntr, int(d / 2), color=255, thickness=-1)

    strel = nd.generate_binary_structure(2, 1).astype(np.uint8)
    er_img = cv2.erode(edimg, strel, iterations=1)
    er_mask = cv2.erode(mask, strel, iterations=1)
    eedimg = edimg - er_img
    eemask = mask - er_mask

    corr = signal.correlate2d(eedimg, eemask)
    corr = corr[d:-(d - 1), d:-(d - 1)]
    corr = corr ** 2
    corr = corr / np.amax(corr)
    corr[corr < corr_thr] = 0

    plt.imshow(corr)
    plt.show()

    # cv2.imshow('corr image', corr.astype('uint8'))
    # cv2.waitKey()

    coords = ftr.peak_local_max(corr, min_distance=5)

    for i, coord in enumerate(coords):
        coords[i, :] = (coord * ds_ratio).astype(dtype=np.uint16)

    for pt in coords:
        imd = cv2.circle(imd, center=(pt[-1], pt[0]), radius=0, color=(0, 0, 255), thickness=10)

    cv2.imshow('origo image with markers', imd.astype('uint8'))
    cv2.waitKey()



    return coords


if __name__ == '__main__':
    from helpers.labeled_jsons import load_labelme_image
    path = '../resources/105mm_60deg.6mt18gqf.000099.json'
    path = os.path.abspath(os.path.join(os.getcwd(), path))
    print('path is: {}'.format(path))
    img = load_labelme_image(path2json=path)

    print('image resolution is {} x {}'.format(img.shape[0], img.shape[1]))

    coords = detect_circles(im=img)
    print(coords)

    from helpers.labeled_jsons import plot_points_on_image
    plot_points_on_image(coords, img)


