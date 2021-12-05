import os
import json
import matplotlib.pyplot as plt
from img_handling.droplets import *
from detector.fringe_count import droplet_slice_from_image, count_fringes
from img_handling.labelme import *

# todo - this histogram comparison doesnt look that good so far, eeeh...

"""Takes the detector json output, gets the number of fringes from the file,
because there is no ground truth for number of fringes, it cuts the gt labels
and uses the same algorithm to check the number of fringes

the test is more for detection of droplets than for the fringe counter,
as the algorithm would test itself against itself:-)"""


DET_OUTPUT_JSON_PATH = '../scripts/det_output.json'
DET_OUTPUT_JSON_PATH = os.path.abspath(DET_OUTPUT_JSON_PATH)


def load_detector_output(path: str):
    # todo - this function is already defined in see_and_eval - centralize it!!! ffs
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print('File {} does not exist'.format(path))
    except Exception as e:
        print('Couldn\'t open file {}, {}'.format(path, e))


data = load_detector_output(DET_OUTPUT_JSON_PATH)


for imname, ostruct in data.items():        # across all images
    dets = ostruct['det']
    gts = ostruct['gt']

    det_fringe_counts = []
    gt_fringe_counts = []

    for det in dets:                        # go across individual droplets
        try:
            if det['fringe_count'] is not None:
                det_fringe_counts.append(det['fringe_count'])
        except Exception as e:
            print('whassaap')
            continue

    for gt in gts:                        # go across individual droplets
        try:
            dl = DropletLabel.init_dict(gt)
            img = load_labelme_image(dl.img_path)
            ds = droplet_slice_from_image(img, dl)
            fr_ct, _, _, _ = count_fringes(ds)
            if fr_ct is not None:
                gt_fringe_counts.append(fr_ct)
        except Exception as e:
            print(e)
            continue

plt.figure()
plt.subplot(121)
plt.title('Ground truth')
plt.xlabel('Number of fringes')
plt.ylabel('counts')
# plt.hist(gt_fringe_counts)
plt.hist(gt_fringe_counts, bins=11, range=(1, 11))
plt.subplot(122)
plt.title('Detected')
plt.xlabel('Number of fringes')
plt.ylabel('counts')
# plt.hist(det_fringe_counts)
plt.hist(det_fringe_counts, bins=11, range=(1, 11))
plt.show()