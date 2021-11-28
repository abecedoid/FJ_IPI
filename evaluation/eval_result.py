import numpy as np
from helpers.labeled_jsons import DropletLabel, load_labelme_image
from scipy.spatial.distance import cdist
import pickle
import warnings
import os


class ConfusionMatrix(dict):
    def __init__(self, gt_labels: list, det_labels: list, gt2dets: dict, from_multiple_imgs: False):
        super(ConfusionMatrix, self).__init__()
        """more or less standard confusion matrix
        inputs:
            - ground truth labels - list of DropletLabel
            - detected labels - list of DropletLabel
            - gt2dets - pairing of gt labels to detected labels

        TP - true positive - detection is where the gt label is
        FP - false positive - detection is where there is no gt label
        TN - true negative - detection is not where it is not supposed to be, eh?
        FN - false negative - detection is missing where the gt label is"""
        self['TP'] = None
        self['FP'] = None
        self['TN'] = None
        self['FN'] = None

        self._gt_labels = gt_labels
        self._det_labels = det_labels
        self._gt2dets = gt2dets

        # get the image path from det labels - checks that all paths are the same
        if not from_multiple_imgs:
            self._impath = None
            for dl in self._det_labels:
                if self._impath is not None:
                    next_impath = dl.img_path
                    if next_impath != self._impath:
                        warnings.warn('List of detection DropletLabels was passed to the ConfusionMatrix'
                                        'with vairous image paths!')
                        self._impath = 'MANY'
                        break
                else:
                    self._impath = dl.img_path
        else:
            self._impath = 'MANY'

        self._compute()

    def _compute(self):
        self['TP'] = len(self._gt2dets.keys())
        self['FP'] = len(self._det_labels) - self['TP']
        self['TN'] = 0  # todo, really??
        self['FN'] = len(self._gt_labels) - self['TP']

    def precision(self):
        return self['TP'] / (self['TP'] + self['FP'])

    def recall(self):
        return self['TP'] / (self['TP'] + self['FN'])

    def sensitivity(self):
        return self['TP'] / (self['TP'] + self['FN'])

    def specificity(self):
        return self['TN'] / (self['TN'] + self['FP'])

    def accuracy(self):
        return (self['TP'] + self['TN']) / len(self._gt_labels)

    # def __add__(self, other):
    #     self['TP'] += other.get('TP')
    #     self['FP'] += other.get('FP')
    #     self['TN'] += other.get('TN')
    #     self['FN'] += other.get('FN')
    #
    #     self._gt_labels = self._gt_labels + other._gt_labels
    #     self._det_labels = self._det_labels + other._det_labels
    #     self._gt2dets.update()

    def __str__(self):
        s = '=======CONFUSION  MATRIX========\n'
        s += 'PATH: {}\n'.format(self._impath)
        s += '|TP: {}\t\t|FN: {}\t\t|\n'.format(self['TP'], self['FN'])
        s += '|FP: {}\t|TN: {}\t\t|\n'.format(self['FP'], self['TN'])
        s += '--------------\n'
        s += 'Sensitivity: {}\n'.format(self.sensitivity())
        s += 'Precision: {}\n'.format(self.precision())
        s += 'Recall: {}\n'.format(self.recall())
        s += 'Precision/Recall: {}\n'.format((self.precision() / self.recall()))
        # s += 'Specificity: {}\n'.format(self.specificity())
        # s += 'Accuracy: {}\n'.format(self.accuracy())
        s += '===============\n'
        return s

    def json(self) -> dict:
        d = {}
        d['img_path'] = self._impath
        d['TP'] = self['TP']
        d['FP'] = self['FP']
        d['TN'] = self['TN']
        d['FN'] = self['FN']

        d['SENSITIVITY'] = self.sensitivity()
        d['PRECISION'] = self.precision()
        d['RECALL'] = self.recall()
        return d


class DetectionEvaluator(object):
    """Takes the ground truth and detections and evaluates the detection performance"""
    def __init__(self, detections: list, labeled: list, from_multiple_imgs=False):
    # def __init__(self, detections: list, labeled: list, image: np.ndarray):
        """both detections and labeled lists are lists of DropletLabels"""
        self._dl_dets = detections
        self._dl_labeled = labeled
        # self._img = image
        # todo - make this dynamic - still problem with the radius... urgh
        self._cradius = 30
        self._from_multiple_imgs = from_multiple_imgs

        self._labels2dets = None    # holds mapping from labels to real detections
        self.confusion_mat = None

    def evaluate(self) -> ConfusionMatrix:
        self._pair_detections_labels()
        self.confusion_mat = ConfusionMatrix(gt_labels=self._dl_labeled,
                                             det_labels=self._dl_dets,
                                             gt2dets=self._labels2dets,
                                             from_multiple_imgs=self._from_multiple_imgs)
        print(self.confusion_mat)
        return self.confusion_mat

    def _pair_detections_labels(self):
        # todo - add score of the detection - basically the distance, right?
        """Pairs up the detections and ground truth labels using the self._labels2dets
        no_of_label -> {"det_id" -> no_of_detection,
                        "dist" -> distance between label and detection} """
        # first get the ndarrays of coordinates for both labels and detections
        det_coords = np.zeros((len(self._dl_dets), 2))
        gt_coords = np.zeros((len(self._dl_labeled), 2))

        for i, det in enumerate(self._dl_dets):
            det_coords[i, :] = det.center()

        for i, gt in enumerate(self._dl_labeled):
            gt_coords[i, :] = gt.center()

        # create distance matrix - distance between center points - rows (labels) X columns (detections)
        distmat = cdist(gt_coords, det_coords)

        # thresholding - distances larger than radius have no overlap, and thus are not interesting
        distmat[distmat >= 2 * self._cradius] = np.inf

        # for each ground truth label, try to find a paired detection
        # if there is none
        gt2det = {}
        for i in range(distmat.shape[0]):

            row_mins = np.nanmin(distmat, axis=1)       # find the minimum distance in each row (minimum distance between label and detection)
            row_min_id = np.argmin(row_mins)            # get the minima's index (row index)
            if np.isinf(row_mins[row_min_id]):          # if the minima is inf, then end search, there are no more pairs...
                break

            col_id = np.argmin(distmat[row_min_id, :])  # find the column id of the detection label
            dist_val = distmat[row_min_id, col_id]

            # disable all the values for the newly-made pair, for both label and detection
            distmat[row_min_id, :] = np.inf
            distmat[:, col_id] = np.inf

            gt2det[row_min_id] = {"det_id": col_id,
                                  "dist": dist_val}

        self._labels2dets = gt2det
        return self._labels2dets

    def stats_print(self):
        """Returns basic stats on how successful the detection was"""
        if self._labels2dets is None:
            return None

        Nlabels2find = len(self._dl_labeled)
        Ndets = len(self._dl_dets)

        Npairs_found = len(self._labels2dets.keys())
        pairs_found_perc = (Npairs_found / Nlabels2find) * 100

        distances = []
        [distances.append(self._labels2dets[entry_key]['dist']) for entry_key in self._labels2dets]
        distances = np.array(distances)

        s = ''
        s += 'GT labels: {}\n'.format(Nlabels2find)
        s += 'Detected labels: {}\n'.format(Ndets)
        s += 'Pairs found: {} ({} %)\n'.format(Npairs_found, pairs_found_perc)

        s += "Distances - mean: {}, std: {}\n".format(distances.mean(), distances.std())

        print(s)


if __name__ == '__main__':
    IMPATH = '../resources/105mm_60deg.6mxcodhz.000000/105mm_60deg.6mxcodhz.000000.json'
    img = load_labelme_image(IMPATH)

    with open('gt_det_droplets.pkl', 'rb') as f:
        dets_dict = pickle.load(f)

    # load the ground truth and detected droplets
    gts = dets_dict['gt']
    dets = dets_dict['det']

    devaluator = DetectionEvaluator(detections=dets, labeled=gts, image=img)
    devaluator.evaluate()
    # devaluator._pair_detections_labels()

    # print(devaluator._labels2dets)
    # devaluator.stats_print()