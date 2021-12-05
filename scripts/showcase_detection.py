import os
import sys
import glob
from img_handling.droplets import *
from img_handling.labelme import *
from detector.circle_detector import detect_circles, preprocess_img
from detector.fringe_count import *
from detector.detector_configuration import *
import json


DIRPATH = 'resources/105mm_60deg.6mxcodhz.000000/'
DIRPATH = os.path.abspath(DIRPATH)

settings = get_detector_settings()
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

LM_PATHS = glob.glob(os.path.join(DIRPATH, '*.json'))
LM_NAMES = [x.split(os.sep)[-1] for x in LM_PATHS]
print(LM_PATHS)
print(LM_NAMES)

det_json = {}

for k, jpath in enumerate(LM_PATHS):
    img = load_labelme_image(jpath)
    gt_droplets = load_labelme_droplet_labels(jpath)

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
    except Exception as e:
        print('Detecting circles in image {} failed: {}, skipping image'.format(jpath, e))
        continue

    # FRINGE COUNT
    for det_droplet in det_droplets:
        try:
            ds = droplet_slice_from_image(img, det_droplet)
            n_fringes, DS, pk_coords, score = count_fringes(ds)
            ds.score = score
            det_droplet.slice = ds
            det_droplet.fringe_count = n_fringes

        except SliceOutOfBoundsError as se:
            print('Slice is out of bounds, cannot count fringes on incomplete circle')
            continue
        # else:
        except Exception as e:
            print('Fringe count failed on a droplet {}, because {}'.format(det_droplet.name, e))
            continue

    # JSONIFY THE OUTPUTS
    gt_droplets_json = [x.json() if x is not None else {} for x in gt_droplets]
    det_droplets_json = [x.json() if x is not None else {} for x in det_droplets]

    det_json[LM_NAMES[k]] = {'gt': gt_droplets_json,
                             'det': det_droplets_json}

with open('scripts/det_output.json', 'w') as f:
    json.dump(det_json, f, indent=4)
# pprint(det_json)


