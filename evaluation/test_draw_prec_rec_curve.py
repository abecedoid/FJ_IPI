import numpy as np
from helpers.labeled_jsons import DropletLabel, load_labelme_image
from scipy.spatial.distance import cdist
import pickle
import os
from evaluation.eval_result import DetectionEvaluator, ConfusionMatrix

if __name__ == '__main__':
    IMPATH = '../resources/105mm_60deg.6mxcodhz.000000/105mm_60deg.6mxcodhz.000000.json'
    img = load_labelme_image(IMPATH)

    with open('gt_det_droplets.pkl', 'rb') as f:
        dets_dict = pickle.load(f)

    # load the ground truth and detected droplets
    gts = dets_dict['gt']
    dets = dets_dict['det']

    devaluator = DetectionEvaluator(detections=dets, labeled=gts, image=img)
    devaluator.evaluate()
    print(devaluator.confusion_mat)