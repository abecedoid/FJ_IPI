import json
import os
from detector.fringe_count import *
from pprint import pprint
from matplotlib import pyplot as plt
from img_handling.droplets import DropletLabel
from img_handling.labelme import load_labelme_image
import numpy as np
from evaluation.evaluator import load_detector_output
from see_fringe_cutouts import plot_multiple_drop_slices


FILEPATH = 'scripts/det_output.json'
FILEPATH = os.path.abspath(FILEPATH)
GT_FRINGE_COUNT = True


def json_output2fringe_key_dict(data: dict) -> dict:
    """Takes the data json data output of the full detector and returns a dict where keys are the number of
    detected fringes"""

    det_fcs = {}

    for imname, ostruct in data.items():  # across all images
        for det in ostruct['det']:  # go across individual droplets

            det_label = DropletLabel.init_dict(det)

            if det_label.fringe_count is None:
                if None not in det_fcs.keys():
                    det_fcs[None] = []
                det_fcs[None].append(det_label)
            else:
                if round(det_label.fringe_count) not in det_fcs.keys():
                    det_fcs[round(det_label.fringe_count)] = []
                det_fcs[round(det_label.fringe_count)].append(det_label)

    return det_fcs


def filter_fringe_key_dict_by_score(fringe_key: dict, interval: tuple) -> dict:
    # todo - this might fail if there are no scores yet...
    """Takes a dictionary of format [number of fringes] -> list[DropletLabel1, ..... N]
    and lets through only droplet labels, which are inside the interval (lower score value, higher score value)
    If either of values in the tuple is None -> that means not bound"""
    # assert interval[0] < interval[1]

    bounds = [*interval]
    if bounds[0] is None:
        bounds[0] = -np.inf
    if bounds[1] is None:
        bounds[0] = np.inf

    out = {}
    for N_fringes, dlabel_list in fringe_key.items():
        out[N_fringes] = []
        for dlabel in dlabel_list:

            if dlabel.score() is None:
                continue

            if bounds[0] <= dlabel.score() <= bounds[1]:
                    out[N_fringes].append(dlabel)

    return out


data = load_detector_output(FILEPATH)

fringe_counts_dets = []
N_dets = 0
N_gts = 0

for imname, ostruct in data.items():        # across all images
    dets = ostruct['det']                   # choose the detection branch
    N_dets += len(dets)
    N_gts += len(ostruct['gt'])
    for det in dets:                        # go across individual droplets

        try:
            if det['fringe_count'] is not None:
                fringe_counts_dets.append(det['fringe_count'])
        except Exception as e:
            print('whassaap')
            continue
N_imgs = len(data.keys())

if GT_FRINGE_COUNT:
    # go through the json output, go across all images, in each take the gts and count the fringes
    fringe_counts_gts = []
    for imname, ostruct, in data.items():
        gts = ostruct['gt']

        img_loaded = False

        for gt in gts:
            # first need to load the image, some gts can be broken and have invalid path
            if not img_loaded:
                if not os.path.exists(gt['img_path']):
                    continue
                img = load_labelme_image(gt['img_path'])
                img_loaded = True

            try:
                droplet = DropletLabel.init_dict(gt)
                ds = droplet_slice_from_image(img, droplet)
                n_fringes, DS, pk_coords, score = count_fringes(ds)
                fringe_counts_gts.append(n_fringes)
            except Exception as e:
                print('Failure when counting fringes of a single droplet: {}'.format(e))


plt.figure()
if GT_FRINGE_COUNT:
    plt.subplot(121)
    plt.hist(fringe_counts_dets, bins=20, range=(1, 20))
    plt.xlabel('number of fringes')
    plt.ylabel('counts')
    plt.title('Detections')
    plt.subplot(122)
    plt.hist(fringe_counts_gts, bins=20, range=(1, 20))
    plt.xlabel('number of fringes')
    plt.ylabel('counts')
    plt.title('Ground truth')
else:
    plt.hist(fringe_counts_dets, bins=20, range=(1, 20))
    plt.xlabel('number of fringes')
    plt.ylabel('counts')
    plt.title('Detections')

plt.show()

dd = json_output2fringe_key_dict(data)
for n_fr in range(3, 20):
    dslices = get_droplet_slices(dd[n_fr])

    plot_multiple_drop_slices(dslices, title='Random sa''mple of slices with {} fringes'.format(n_fr))


# todo - when scoring system is done
# split into good and bad based on score...
# dx_bad = filter_fringe_key_dict_by_score(dd, interval=(None, 50))
# dx_good = filter_fringe_key_dict_by_score(dd, interval=(20, None))

# dslices_bad = get_droplet_slices(dx_bad[4])
# dslices_good = get_droplet_slices(dx_good[4])



