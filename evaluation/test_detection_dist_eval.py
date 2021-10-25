import pickle
import numpy as np
from scipy.spatial.distance import cdist
from helpers.labeled_jsons import plot_multiple_droplet_lists_on_image, load_labelme_image
from operator import itemgetter
import os


with open('gt_det_droplets.pkl', 'rb') as f:
    dets_dict = pickle.load(f)

# load the ground truth droplets
gts = dets_dict['gt']
# load the detected droplets
dets = dets_dict['det']

# compute the distance matrix [rows - ground truth, columns - detections]
det_coords = np.zeros((len(dets), 2))
gt_coords = np.zeros((len(gts), 2))

for i, det in enumerate(dets):
    det_coords[i, :] = det.center()

for i, gt in enumerate(gts):
    gt_coords[i, :] = gt.center()

distmat = cdist(gt_coords, det_coords)

# all distances larger than radius - set to nan
cradius = 30
distmat[distmat >= 2*cradius] = np.inf

# find rows fully nan - these ground truths have no detected pair...
# isrow_fully_nan = np.all(np.isnan(distmat), axis=1)

gt2det = {}
for i in range(distmat.shape[0]):
    row_mins = np.nanmin(distmat, axis=1)
    row_id = np.argmin(row_mins)
    if np.isinf(row_mins[row_id]):
        break
    # get the position of the minimum column-wise
    col_id = np.argmin(distmat[row_id, :])
    distmat[row_id, :] = np.inf
    distmat[:, col_id] = np.inf
    gt2det[row_id] = col_id

print(gt2det)
print('n of values: {}'.format(len(gt2det.values())))
print('n of unique values: {}'.format(len(set(gt2det.values()))))

# make a set of paired detections
pdets = itemgetter(*list(gt2det.values()))(dets)

plot_dict = {'gets': gts, 'dets': pdets}

DIRPATH = '../resources/105mm_60deg.6mxcodhz.000000'
DIRPATH = os.path.abspath(DIRPATH)
imfnames = os.listdir(DIRPATH)

for imfname in imfnames:
    impath = os.path.join(DIRPATH, imfname)
    img = load_labelme_image(impath)

    plot_multiple_droplet_lists_on_image(plot_dict, img)
    break