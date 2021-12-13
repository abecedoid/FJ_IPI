import numpy as np
from img_handling.labelme import load_labelme_image
from scipy.spatial.distance import cdist
import warnings
import json
from img_handling.droplets import DropletLabel
from img_handling.plotters import plot_multiple_droplet_lists_on_image
import os
import cv2
from detector.detector_configuration import get_evaluation_settings
from evaluation.evaluator import *


if __name__ == '__main__':

    OUTPUT_JSON_FILEPATH = 'all_resource_detections.json'
    OUTPUT_JSON_FILEPATH = os.path.abspath(OUTPUT_JSON_FILEPATH)
    SHOW_IMAGES = True

    data = load_detector_output(OUTPUT_JSON_FILEPATH)

    fringe_counts = []
    N_dets = 0
    N_gts = 0
    conf_mats = {}
    all_det_dls = []
    all_gt_dls = []

    for imname, ostruct in data.items():        # across all images
        dets = ostruct['det']
        gts = ostruct['gt']

        det_dls = json_det_structs2droplet_list(dets)
        gt_dls = json_det_structs2droplet_list(gts)

        conf_mats[imname] = get_json_evaluation(det_dls, gt_dls)

        # add to list with all detections and gts over all directory
        all_det_dls = all_det_dls + det_dls
        all_gt_dls = all_gt_dls + gt_dls

        if SHOW_IMAGES:
            # load the image
            for det in dets:
                dl = DropletLabel.init_dict(det)
                if os.path.exists(dl.img_path):
                    impath = dl.img_path
                    break
            img = load_labelme_image(impath)

            # show the image
            plot_multiple_droplet_lists_on_image({'gts': gt_dls, 'dets': det_dls}, img=img, wait_key=False)
            if cv2.waitKey(0) == 27:
                SHOW_IMAGES = False

    # evaluation across all images in directory
    master_conf_mat = get_json_evaluation(all_det_dls, all_gt_dls, from_multiple_imgs=True)

    json_output = {'overall_evaluation': master_conf_mat.json(),
                   'individual_evaluations': [x.json() for key, x in conf_mats.items()]}
    with open('all_resource_evaluation.json', 'w') as f:
        json.dump(json_output, f, indent=4)

