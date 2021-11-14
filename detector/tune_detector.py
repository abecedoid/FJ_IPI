import numpy as np
from helpers.labeled_jsons import DropletLabel, load_labelme_image, coords2droplet_labels_list, load_labelme_droplet_labels
from helpers.labeled_jsons import plot_multiple_droplet_lists_on_image
from scipy.spatial.distance import cdist
import pickle
import os
from detector.circle_detector import detect_circles_labelme_input, detect_circles
from evaluation.eval_result import DetectionEvaluator, ConfusionMatrix

dsettings_normal = {
    'ds_coeff': 2,
    'circle_mask_rad': 30,
    'circle_mask_wdth': None,
    'circle_mask_radoff_size': 5,
    'pxcorr1': 80,
    'pxcorr2': 80,
    'peakfind_thr': 0.1,
    'peakfind_min_max_nghbr': 30,
    'debug': True
}

if __name__ == '__main__':
    s = dsettings_normal
    IMPATH = '../resources/105mm_60deg.6mxcodhz.000000/105mm_60deg.6mxcodhz.000000.json'
    img = load_labelme_image(IMPATH)
    gt_droplets = load_labelme_droplet_labels(IMPATH)

    # img, coords = detect_circles_labelme_input(img_path=IMPATH)
    img = load_labelme_image(IMPATH)
    coords = detect_circles(img, DS_COEFF=s['ds_coeff'],
                            circle_mask_rad=s['circle_mask_rad'],
                            circle_mask_wdth=s['circle_mask_wdth'],
                            circle_mask_radoff_size=s['circle_mask_radoff_size'],
                            pxcorr1=s['pxcorr1'], pxcorr2=s['pxcorr2'],
                            peakfind_thr=s['peakfind_thr'],
                            peakfind_min_max_nghbr=s['peakfind_min_max_nghbr'],
                            debug=s['debug'])

    det_droplets = coords2droplet_labels_list(coords, circle_radius=30)

    devaluator = DetectionEvaluator(detections=det_droplets, labeled=gt_droplets, image=img)
    devaluator.evaluate()
    print(devaluator.confusion_mat)

    plot_dict = {'gt': gt_droplets, 'det': det_droplets}
    plot_multiple_droplet_lists_on_image(plot_dict, img)
