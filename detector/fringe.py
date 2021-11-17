import numpy as np
from helpers.labeled_jsons import DropletLabel, load_labelme_image, DropletSlice
from detector.circle_detector import find_peaks2d


# class DropletSlice(object):
#     def __init__(self, img: np.ndarray, radius: int, score=None):
#         self._img = img
#         self._radius = radius
#         self._score = score
#
#     @property
#     def img(self):
#         return self._img
#
#     @img.setter
#     def img(self, img: np.ndarray):
#         self._img = img
#
#     @property
#     def radius(self):
#         return self._radius
#
#     def center_coords(self) -> list:
#         return [int(self._img.shape[0]/2), int(self._img.shape[1]/2)]
#
#     def rotate2def(self):
#         raise NotImplemented


def get_droplet_slices_from_img(img: np.ndarray, droplet_labels: list) -> list:
    """Returns a list of droplet slices from one image"""
    dslices = []
    for dlabel in droplet_labels:
        try:
            dslices.append(droplet_slice_from_image(img, dlabel))
        except Exception as e:
            print('Failed to get droplet slice from img at coords {}, cause: {}'.format(str(DropletLabel), e))


def get_droplet_slices(droplet_labels: list) -> list:
    """Returns a list of droplet slices from any images if accessible"""
    dslices = []
    # fill empty array with data
    for k, dlabel in enumerate(droplet_labels):

        img = load_labelme_image(path2json=dlabel.img_path)

        try:
            dslice = droplet_slice_from_image(img, dlabel)
        except SliceOutOfBoundsError as serr:
            print('slice out of bounds...')
            dslices.append(None)
            continue
        except Exception as e:
            print('some other problem with getting the slice {}'.format(e))
            dslices.append(None)
            continue

        dslices.append(dslice)
    return dslices


def droplet_slice_from_image(img: np.ndarray, droplet_label: DropletLabel, radius_offset: int = 5) -> DropletSlice:
    # compute slice's side
    side = 2 * droplet_label.radius() + 2 * radius_offset

    # lower x coord
    lx = int(droplet_label.center()[1] - side/2)
    # lower y coord
    ly = int(droplet_label.center()[0] - side/2)

    # higher x coord
    hx = int(droplet_label.center()[1] + side / 2)
    # higher y coord
    hy = int(droplet_label.center()[0] + side / 2)

    if lx < 0 or ly < 0 or hx > img.shape[0] or hy > img.shape[1]:
        raise SliceOutOfBoundsError

    img_slice = img[lx: lx + side, ly: ly + side]

    return DropletSlice(img_slice, droplet_label.radius(), score=None)


class SliceOutOfBoundsError(Exception):
    pass


def dist2points(p1: list, p2: list) -> float:
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def count_fringes(ds: DropletSlice, max_min_fltr_size: int=5, peak_thr: float=0.1) \
        -> (float, np.ndarray, np.ndarray, float):
    """Takes the image and estimates number of fringes in it
    Returns the number of fringes, the FFT(img) and the coords of the FFT(img) peaks (excluding the DC part)"""
    # todo - if/when some time, solve it for the occasion, when two droplets overlap (multiple maxima in fft and all that jazz
    # fast fourie it
    I = np.abs(np.fft.fftshift(np.fft.fft2(ds.img)))
    # normalize it for thresholding in find 2d peaks
    I = I / np.max(I[:])

    # find peaks
    pk_coords = find_peaks2d(arr=I, max_min_neighborhood_size=max_min_fltr_size, threshold=peak_thr)

    # compute the ratio of value from peaks to background
    pk_vals = np.zeros((pk_coords.shape[0]))
    I_cpy = I.copy()
    for k in range(pk_vals.shape[0]):

        pk_vals[k] = I[int(pk_coords[k, 0]), int(pk_coords[k, 1])]
        I_cpy[int(pk_coords[k, 0]), int(pk_coords[k, 1])] = np.nan
    print('hehe')
    pk_mval = np.nanmean(pk_vals)
    bgrnd = np.nanmean(I_cpy[:])
    score = pk_mval / bgrnd
    if pk_coords.shape[0] != 3:
        score = 0
    print('score:{}'.format(score))

    # from scipy.stats import skew
    # sk = skew(I, axis=None)
    # print('skewness is {}'.format(sk))

    # histogram of values
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(I)
    # plt.subplot(122)
    # plt.hist(I[:])
    # plt.title('Ratio (score) is {}'.format(score))
    # plt.show()

    # remove the center DC coordinate
    try:
        # find row with the center index
        row_idx = np.where((pk_coords == ds.center_coords()).all(axis=1))[0][0]
        np.delete(pk_coords, row_idx, axis=0)
    except Exception as e:
        # print('for some reason the center was not a part of detected peaks: {}'.format(e))
        print('Fringe count failed, skipping...: {}'.format(e))

    # compute the distance between center and peak
    dist = dist2points(ds.center_coords(), pk_coords[0])
    return dist, I, pk_coords, score
