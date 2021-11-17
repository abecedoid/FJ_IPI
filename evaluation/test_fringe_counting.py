import os
import numpy as np
from helpers.labeled_jsons import *
import cv2
from detector.fringe import *
import matplotlib.pyplot as plt
from detector.circle_detector import find_peaks2d



if __name__ == '__main__':

    DIRPATH = '../resources/105mm_60deg.6mxcodhz.000000'
    DIRPATH = os.path.abspath(DIRPATH)
    imfnames = os.listdir(DIRPATH)

    for imfname in imfnames:

        impath = os.path.join(DIRPATH, imfname)

        # load numpy image and ground truth droplets
        img = load_labelme_image(impath)
        gt_droplets = load_labelme_droplet_labels(impath)

        for gt_droplet in gt_droplets:

            if gt_droplet is None:
                continue

            ds = droplet_slice_from_image(img, gt_droplet)

            # random rotation
            for i in range(int(np.round(np.random.random() * 3))):
                ds.img = np.rot90(ds.img)

            n, DS, pk_coords = count_fringes(ds)
            print('fringe count is: {}'.format(n))

            plt.figure()
            plt.subplot(131)
            plt.imshow(ds.img)
            plt.subplot(132)
            plt.imshow(DS)
            plt.subplot(133)
            plt.imshow(DS)
            plt.scatter(pk_coords[:, 0], pk_coords[:, 1], color='r', s=1)
            plt.show()




