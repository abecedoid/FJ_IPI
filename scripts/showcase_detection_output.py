import json
import os
from detector.fringe import droplet_slice_from_image, SliceOutOfBoundsError, get_droplet_slices_from_img, get_droplet_slices
from pprint import pprint
from matplotlib import pyplot as plt
from helpers.labeled_jsons import DropletLabel, load_labelme_image
import numpy as np


FILEPATH = 'det_output.json'
FILEPATH = os.path.abspath(FILEPATH)


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


def plot_multiple_drop_slices(dslices: list):
    MAX_PLOTS = 91
    if len(dslices) > MAX_PLOTS:
        print('MAX PLOT is set to {}, not showing all {} slices'.format(MAX_PLOTS, len(dslices)))
        dslices = dslices[:MAX_PLOTS]

    plt.figure()
    idx = 0
    for m in range(1, 9):
        for n in range(1, 9):
            idx += 1
            if idx > len(dslices):
                break
            try:
                plt.subplot(9, 9, idx)
                plt.imshow(dslices[idx].img)
            except Exception as e:
                break

    plt.show()


try:
    with open(FILEPATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print('File {} does not exist'.format(FILEPATH))
except Exception as e:
    print('Couldn\'t open file {}, {}'.format(FILEPATH, e))

fringe_counts = []
N_dets = 0
N_gts = 0

for imname, ostruct in data.items():        # across all images
    dets = ostruct['det']                   # choose the detection branch
    N_dets += len(dets)
    N_gts += len(ostruct['gt'])
    for det in dets:                        # go across individual droplets

        try:
            if det['fringe_count'] is not None:
                    fringe_counts.append(det['fringe_count'])
        except Exception as e:
            print('whassaap')
            continue
N_imgs = len(data.keys())

N_ok_fringes = len(fringe_counts)
print('output is from {} images'.format(N_imgs))
print('detections: {}/ground truth: {}'.format(N_dets, N_gts))
print('{} Fringes extracted from {} detections ({}/{})'.format(N_ok_fringes, N_dets,
                                                               N_ok_fringes, N_dets))
print('we have {} fringe counts'.format(len(fringe_counts)))
plt.figure()
plt.hist(fringe_counts, bins=50)
plt.show()


dd = json_output2fringe_key_dict(data)
dslices = get_droplet_slices(dd[4])
plot_multiple_drop_slices(dslices)

print('hehe')

