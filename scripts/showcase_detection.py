import os
import glob
from helpers.labeled_jsons import *
from detector.circle_detector import detect_circles
from detector.fringe import *
import json
from pprint import pprint


dsettings = {
    'ds_coeff': 2,
    'circle_mask_rad': 30,
    'circle_mask_wdth': None,
    'circle_mask_radoff_size': 5,
    'pxcorr1': 90,
    'pxcorr2': 90,
    'peakfind_thr': 0.1,
    'peakfind_min_max_nghbr': 30,
    'debug': False
}

DIRPATH = '../resources/105mm_60deg.6mxcodhz.000000/'
DIRPATH = os.path.abspath(DIRPATH)

LM_PATHS = glob.glob(os.path.join(DIRPATH, '*.json'))
LM_NAMES = [x.split(os.sep)[-1] for x in LM_PATHS]
print(LM_PATHS)
print(LM_NAMES)

det_json = {}

for k, jpath in enumerate(LM_PATHS):
    img = load_labelme_image(jpath)
    gt_droplets = load_labelme_droplet_labels(jpath)

    def preprocess_img(img: np.ndarray):
        # img = cv2.normalize(img, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(60, 60))
        img = clahe.apply(img)
        img[img < 50] = 0
        return img

    pimg = preprocess_img(img)

    try:
        coords = detect_circles(pimg, DS_COEFF=dsettings['ds_coeff'],
                                circle_mask_rad=dsettings['circle_mask_rad'],
                                circle_mask_wdth=dsettings['circle_mask_wdth'],
                                circle_mask_radoff_size=dsettings['circle_mask_radoff_size'],
                                pxcorr1=dsettings['pxcorr1'], pxcorr2=dsettings['pxcorr2'],
                                peakfind_thr=dsettings['peakfind_thr'],
                                peakfind_min_max_nghbr=dsettings['peakfind_min_max_nghbr'],
                                debug=dsettings['debug'])

        det_droplets = coords2droplet_labels_list(coords, circle_radius=dsettings['circle_mask_rad'],
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

with open('det_output.json', 'w') as f:
    json.dump(det_json, f, indent=4)
# pprint(det_json)


