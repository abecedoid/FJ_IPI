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

OUTPUT_JSON_FILEPATH = '../scripts/det_output.json'
OUTPUT_JSON_FILEPATH = os.path.abspath(OUTPUT_JSON_FILEPATH)


try:
    with open(OUTPUT_JSON_FILEPATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print('File {} does not exist'.format(OUTPUT_JSON_FILEPATH))
except Exception as e:
    print('Couldn\'t open file {}, {}'.format(OUTPUT_JSON_FILEPATH, e))


fringe_counts = []
N_dets = 0
N_gts = 0

for imname, ostruct in data.items():        # across all images
    dets = ostruct['det']
    gts = ostruct['gt']
    N_dets += len(dets)
    N_gts += len(ostruct['gt'])

    # load the image
    for det in dets:
        dl = DropletLabel.init_dict(det)
        if os.path.exists(dl.img_path):
            impath = dl.img_path
    img = load_labelme_image(impath)

    det_dls = []
    # load all det dropletlables
    for det in dets:                        # go across individual droplets
        try:
            det_dls.append(DropletLabel.init_dict(det))
        except Exception as e:
            print('failed to initialize droplet label from json output')
            continue

    gt_dls = []
    # load all gt droplets
    for gt in gts:                        # go across individual droplets
        try:
            gt_dls.append(DropletLabel.init_dict(gt))
        except Exception as e:
            print('failed to initialize droplet label from json output')
            continue

    # show the image
    plot_multiple_droplet_lists_on_image({'dets': det_dls, 'gts': gt_dls}, img = img)



# N_imgs = len(data.keys())



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




