import os
import numpy as np
import math
from detector.circle_detector import detect_circles_labelme_input, detect_circles
from detector.fringe import get_droplet_slices_from_img
from helpers.labeled_jsons import *
import matplotlib.pyplot as plt
import pickle


def two_circle_area_intersection(c1: DropletLabel, c2: DropletLabel) -> float:
    """Finds intersection area of two circles.
    Returns intersection area of two circles otherwise 0
    """

    d = math.dist(c1.center(), c2.center())
    rad1sqr = c1.radius() ** 2
    rad2sqr = c2.radius() ** 2

    if d == 0:
        # the circle centers are the same
        return math.pi * min(c1.radius(), c2.radius()) ** 2

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1.radius() * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2.radius() * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = math.acos(angle1) * 2
        theta2 = math.acos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))

        return area1 + area2
    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return math.pi * min(c1.radius(), c2.radius()) ** 2
    return 0


def circ_intersection_over_union(truth: DropletLabel, det: DropletLabel):
    """Computes the intersection over union for two DropletLabels"""
    # compute intersection
    intersection = two_circle_area_intersection(truth, det)
    # compute union
    union = truth.area() + det.area()
    # intersection over union
    iou = intersection / union
    return iou


s = {
    'ds_coeff': 2,
    'circle_mask_rad': 30,
    'circle_mask_wdth': None,
    'circle_mask_radoff_size': 5,
    'pxcorr1': 95,
    'pxcorr2': 95,
    'peakfind_thr': 0.1,
    'peakfind_min_max_nghbr': 30,
    'debug': False
}


if __name__ == '__main__':

    SHOW_IMAGES = True
    DIRPATH = '../resources/105mm_60deg.6mxcodhz.000000'
    DIRPATH = os.path.abspath(DIRPATH)
    imfnames = os.listdir(DIRPATH)

    for imfname in imfnames:

        impath = os.path.join(DIRPATH, imfname)

        img = load_labelme_image(impath)
        gt_droplets = load_labelme_droplet_labels(impath)

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img)

        # PREPROCESSING - JUST FOR NOW
        # bckg_thr = 20
        # alpha = 1.8
        # beta = 40
        # img[img < bckg_thr] = 0
        # # img = cv2.equalizeHist(img)
        # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # plt.subplot(122)
        # plt.imshow(img)
        # plt.show()

        # coords = detect_circles(img, debug=True, circle_mask_rad=20)
        coords = detect_circles(img, DS_COEFF=s['ds_coeff'],
                                circle_mask_rad=s['circle_mask_rad'],
                                circle_mask_wdth=s['circle_mask_wdth'],
                                circle_mask_radoff_size=s['circle_mask_radoff_size'],
                                pxcorr1=s['pxcorr1'], pxcorr2=s['pxcorr2'],
                                peakfind_thr=s['peakfind_thr'],
                                peakfind_min_max_nghbr=s['peakfind_min_max_nghbr'],
                                debug=s['debug'])
        print(coords)
        ious = []
        det_droplets = []
        for gt_drop in gt_droplets:

            # make coords to dropletLabels
            for i, coord in enumerate(coords):
                try:
                    name = 'detection_{}'.format(i)
                    det_drop = DropletLabel(center_pt=coord, radius=30, name=name)
                    det_droplets.append(det_drop)
                    ious.append(circ_intersection_over_union(truth=gt_drop, det=det_drop))
                except Exception as e:
                    continue

        plot_dict = {'gt': gt_droplets, 'det': det_droplets}

        with open('gt_det_droplets.pkl', 'wb') as f:
            pickle.dump(plot_dict, f)

        print('sum of ious: {}'.format(sum(ious)))
        plot_multiple_droplet_lists_on_image(plot_dict, img)

        # plt.figure()
        # plt.plot(ious)
        # plt.show()


    # SHOW_IMAGES = True
    # PATH = '..//resources//105mm_60deg.6mt18gqf.000099.json'
    # img = load_labelme_image(PATH)
    # gt_droplets = load_labelme_droplet_labels(PATH)
    # coords = detect_circles(img)
    #
    # ious = []
    # det_droplets = []
    # for gt_drop in gt_droplets:
    #
    #     # make coords to dropletLabels
    #     for i, coord in enumerate(coords):
    #         name = 'detection_{}'.format(i)
    #         det_drop = DropletLabel(center_pt=coord, radius=52, name=name)
    #         det_droplets.append(det_drop)
    #         ious.append(circ_intersection_over_union(truth=gt_drop, det=det_drop))
    #
    # plot_dict = {'gt': gt_droplets, 'det': det_droplets}
    # plot_multiple_droplet_lists_on_image(plot_dict, img)
    #
    # plt.figure()
    # plt.plot(ious)
    # plt.show()
    # print('sum of ious: {}'.format(sum(ious)))

