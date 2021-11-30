from pprint import pprint
from detector.circle_detector import detect_circles_labelme_input, detect_circles, preprocess_img
from evaluation.eval_result import DetectionEvaluator, ConfusionMatrix
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from helpers.labeled_jsons import *
from evaluation.see_and_eval import *

## SO LOOKS LIKE THESE PARAMETERS ARE OK:
# settings = {
#     # not need to tune...
#     'circle_mask_rad': 30,      # this should be the radius of the circl in px
#     'circle_mask_radoff_size': 5,
#     'ds_coeff': 2,
#     'circle_mask_wdth': None,
#     'debug': False,
#     # tune these
#     'pxcorr1': 94,                  # tune from 80 to 99
#     'pxcorr2': 93,                  # tune from 80 to 99
#     'peakfind_thr': 0.2,          # tune from 0.001 to 0.5
#     'peakfind_min_max_nghbr': 20,   # tune from 10 to 50
#     'CLAHE_clip_limit': 22,          # tune from 2 to 50
#     'CLAHE_grid_size': 13            # tune from 8 to 60
# }


settings = {
    # not need to tune...
    'circle_mask_rad': 30,      # this should be the radius of the circl in px
    'circle_mask_radoff_size': 5,
    'ds_coeff': 2,
    'circle_mask_wdth': None,
    'debug': False,
    # tune these
    'pxcorr1': 94,                  # tune from 80 to 99
    'pxcorr2': 93,                  # tune from 80 to 99
    'peakfind_thr': 0.2,          # tune from 0.001 to 0.5
    'peakfind_min_max_nghbr': 5,   # tune from 10 to 50
    'CLAHE_clip_limit': 22,          # tune from 2 to 50
    'CLAHE_grid_size': 13            # tune from 8 to 60
}

TUNE_PARAMETER = 'peakfind_min_max_nghbr'
TUNE_N_STEPS = 10
TUNE_STEP = 5

DIRPATH = '../../resources/105mm_60deg.6mxcodhz.000000/'
DIRPATH = os.path.abspath(DIRPATH)

LM_PATHS = glob.glob(os.path.join(DIRPATH, '*.json'))
LM_NAMES = [x.split(os.sep)[-1] for x in LM_PATHS]

param2eval = {}
precisions = []
recalls = []
param_vals = []

for tstep in range(TUNE_N_STEPS):

    settings[TUNE_PARAMETER] += TUNE_STEP       # set new parameter value for detector
    param_vals.append(settings[TUNE_PARAMETER])
    all_det_dls = []
    all_gt_dls = []

    for k, jpath in enumerate(LM_PATHS):
        img = load_labelme_image(jpath)
        gt_droplets = load_labelme_droplet_labels(jpath)
        all_gt_dls.extend(gt_droplets)

        pimg = preprocess_img(img, settings['CLAHE_clip_limit'], settings['CLAHE_grid_size'])

        try:
            coords = detect_circles(pimg, DS_COEFF=settings['ds_coeff'],
                                    circle_mask_rad=settings['circle_mask_rad'],
                                    circle_mask_wdth=settings['circle_mask_wdth'],
                                    circle_mask_radoff_size=settings['circle_mask_radoff_size'],
                                    pxcorr1=settings['pxcorr1'], pxcorr2=settings['pxcorr2'],
                                    peakfind_thr=settings['peakfind_thr'],
                                    peakfind_min_max_nghbr=settings['peakfind_min_max_nghbr'],
                                    debug=settings['debug'])

            det_droplets = coords2droplet_labels_list(coords, circle_radius=settings['circle_mask_rad'],
                                                      img_path=jpath)
            all_det_dls.extend(det_droplets)
        except Exception as e:
            print('Detecting circles in image {} failed: {}, skipping image'.format(jpath, e))
            continue

    master_conf_mat = get_json_evaluation(all_det_dls, all_gt_dls, from_multiple_imgs=True)
    precisions.append(master_conf_mat.precision())
    recalls.append(master_conf_mat.recall())
    print(master_conf_mat)

plt.figure()
plt.title('Tune of {}'.format(TUNE_PARAMETER))
plt.xlabel('recall')
plt.ylabel('precision')
plt.plot(recalls, precisions, 'bo-')

i = 0
for x, y in zip(recalls, precisions):
    label = param_vals[i]
    plt.annotate(label, (x, y),
                 textcoords='offset points',
                 ha='center',
                 xytext=(0, 10))
    i += 1

plt.show()

