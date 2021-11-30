import json
import os
from detector.fringe import droplet_slice_from_image, SliceOutOfBoundsError, get_droplet_slices_from_img, get_droplet_slices
from pprint import pprint
from matplotlib import pyplot as plt
from helpers.labeled_jsons import DropletLabel, load_labelme_image
import numpy as np
import random


FILEPATH = 'det_output.json'
FILEPATH = os.path.abspath(FILEPATH)
NO_FRINGES = 4


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


def plot_multiple_drop_slices(dslices: list, title: str = 'title'):
    MAX_PLOTS = 100
    if len(dslices) < MAX_PLOTS:
        N_PLOTS = len(dslices)
        d2plot = dslices
    else:           # make random selection
        N_PLOTS = MAX_PLOTS
        d2plot = random.sample(dslices, N_PLOTS)

    plt.figure()
    idx = 1
    breakaway = False
    for m in range(1, 11):

        if breakaway:
            break

        for n in range(1, 11):
            plt.subplot(10, 10, idx)
            plt.imshow(d2plot[idx].img)
            idx += 1
            if idx >= N_PLOTS:
                breakaway = True
                break

    plt.suptitle(title)
    plt.show()


try:
    with open(FILEPATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print('File {} does not exist'.format(FILEPATH))
except Exception as e:
    print('Couldn\'t open file {}, {}'.format(FILEPATH, e))


dd = json_output2fringe_key_dict(data)
dx = dd[NO_FRINGES]
dslices = get_droplet_slices(dd[NO_FRINGES])
plot_multiple_drop_slices(dslices, title='Cutouts with {} fringes'.format(NO_FRINGES))