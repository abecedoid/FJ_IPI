import os
import sys
sys.path.append('../')
from img_handling.labelme import *
from detector.fringe_count import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    DIRPATH = '../../resources/105mm_60deg.6mxcodhz.000000'
    DIRPATH = os.path.abspath(DIRPATH)
    imfnames = os.listdir(DIRPATH)

    for imfname in imfnames:

        impath = os.path.join(DIRPATH, imfname)

        # load numpy image and ground truth droplets
        img = load_labelme_image(impath)
        gt_droplets = load_labelme_droplet_labels(impath)

        for gt_droplet in gt_droplets:
            try:
                ds = droplet_slice_from_image(img, gt_droplet)

                plt.figure()
                plt.imshow(ds.img)
                plt.show()
            except Exception as e:
                continue


