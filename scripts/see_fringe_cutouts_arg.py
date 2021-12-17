import os
from detector.fringe_count import get_droplet_slices
from matplotlib import pyplot as plt
from img_handling.labelme import DropletLabel
import random
from evaluation.evaluator import load_detector_output
import argparse
import sys

"""This script takes the detector output and shows a random selection of droplet cutouts with 
defined number of detected fringes"""


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


if __name__ == '__main__':

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='This script takes the detector output and shows a \
                                     random selection of droplet cutouts with defined number of detected fringes')
    parser.add_argument('-i', '--inp', help='path to the detector output json file ')
    parser.add_argument('-n', '--nfringes', help='number of fringes which you want to inspect (values <3 - 15>')
    args = vars(parser.parse_args())

    FILEPATH = os.path.abspath(args['inp'])
    if not os.path.exists(FILEPATH):
        print('Please specify an existing file, specified now: {}'.format(FILEPATH))
        sys.exit()

    try:
        NO_FRINGES = int(args['nfringes'])
        if NO_FRINGES < 3 or NO_FRINGES > 15:
            print('Please specify integer in the interval <3 - 15>, specified: {}'.format(args['nfringes']))
            sys.exit()
    except ValueError:
        print('Invalid argument for nfringes: {}'.format(args['nfringes']))
        sys.exit()

    # SCRIPT ITSELF
    data = load_detector_output(FILEPATH)
    dd = json_output2fringe_key_dict(data)
    dx = dd[NO_FRINGES]
    dslices = get_droplet_slices(dd[NO_FRINGES])
    plot_multiple_drop_slices(dslices, title='Cutouts with {} fringes'.format(NO_FRINGES))

