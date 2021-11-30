import os
import sys
import cv2

import numpy as np
from helpers.labeled_jsons import load_labelme_image, plot_image, \
    load_labelme_droplet_labels, plot_multiple_droplet_lists_on_image
import matplotlib.pyplot as plt
import glob
from detector.circle_detector import detect_circles
from detector.fringe import droplet_slice_from_image
import json
from helpers.labeled_jsons import DropletLabel, DropletSlice
from evaluation.eval_result import DetectionEvaluator, ConfusionMatrix


OUTPUT_JSON_FILEPATH = '../scripts/det_output.json'
OUTPUT_JSON_FILEPATH = os.path.abspath(OUTPUT_JSON_FILEPATH)
SHOW_IMAGES = True


def load_detector_output(path: str):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print('File {} does not exist'.format(path))
    except Exception as e:
        print('Couldn\'t open file {}, {}'.format(path, e))


def json_det_structs2droplet_list(json_structs: list):
    droplets = []
    for struct in json_structs:
        try:
            droplets.append(DropletLabel.init_dict(struct))
        except Exception as e:
            print('failed to initialize droplet label from json output: {}'.format(e))
            continue
    return droplets


def get_json_evaluation(det_dls: list, gt_dls: list, from_multiple_imgs=False) -> ConfusionMatrix:
    """Wrapper function
    det_dls - list of detection DropletLabels
    gt_dls - list of ground truth DropletLabels"""
    det_evaluator = DetectionEvaluator(detections=det_dls, labeled=gt_dls, from_multiple_imgs=from_multiple_imgs)
    conf_mat = det_evaluator.evaluate()
    return conf_mat


if __name__ == '__main__':
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
            plot_multiple_droplet_lists_on_image({'dets': det_dls, 'gts': gt_dls}, img=img, wait_key=False)
            if cv2.waitKey(0) == 27:
                SHOW_IMAGES = False


    # evaluation across all images in directory
    master_conf_mat = get_json_evaluation(all_det_dls, all_gt_dls, from_multiple_imgs=True)

    json_output = {'overall_evaluation': master_conf_mat.json(),
                   'individual_evaluations': [x.json() for key, x in conf_mats.items()]}
    with open('det_evaluation.json', 'w') as f:
        json.dump(json_output, f, indent=4)



# plt.figure()
# plt.subplot(221)
# plt.imshow(img)
# plt.title('original')
# plt.subplot(223)
# plt.hist(img, bins=256 ,histtype='step')

# plt.subplot(222)
# plt.imshow(pimg)
# plt.title('processed')
# plt.subplot(224)
# plt.hist(pimg, bins=256, histtype='step')
# plt.show()




