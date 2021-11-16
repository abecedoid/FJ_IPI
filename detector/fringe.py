import numpy as np
from helpers.labeled_jsons import DropletLabel
from detector.circle_detector import find_peaks2d


class DropletSlice(object):
    def __init__(self, img: np.ndarray, radius: int, score=None):
        self._img = img
        self._radius = radius
        self._score = score

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img: np.ndarray):
        self._img = img

    @property
    def radius(self):
        return self._radius

    def center_coords(self) -> list:
        return [int(self._img.shape[0]/2), int(self._img.shape[1]/2)]

    def rotate2def(self):
        raise NotImplemented


def get_droplet_slices_from_img(img: np.ndarray, droplet_labels: list) -> list:
    """Returns a list of droplet slices"""
    dslices = []
    for dlabel in droplet_labels:
        try:
            dslices.append(droplet_slice_from_image(img, dlabel))
        except Exception as e:
            print('Failed to get droplet slice from img at coords {}, cause: {}'.format(str(DropletLabel), e))


def droplet_slice_from_image(img:np.ndarray, droplet_label: DropletLabel, radius_offset: int=5) -> DropletSlice:
    # compute slice's side
    side = 2 * droplet_label.radius() + 2 * radius_offset
    # lower x coord
    lx = int(droplet_label.center()[1] - side/2)
    # lower y coord
    ly = int(droplet_label.center()[0] - side/2)

    if lx < 0 or ly < 0:
        raise SliceOutOfBoundsError

    img_slice = img[lx: lx + side, ly: ly + side]

    return DropletSlice(img_slice, droplet_label.radius(), score=None)


class SliceOutOfBoundsError(Exception):
    pass


def dist2points(p1: list, p2: list) -> float:
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def count_fringes(ds: DropletSlice, max_min_fltr_size: int=3, peak_thr: float=0.1) \
        -> (float, np.ndarray, np.ndarray):
    """Takes the image and estimates number of fringes in it
    Returns the number of fringes, the FFT(img) and the coords of the FFT(img) peaks (excluding the DC part)"""
    # todo - if/when some time, solve it for the occasion, when two droplets overlap (multiple maxima in fft and all that jazz
    # fast fourie it
    I = np.abs(np.fft.fftshift(np.fft.fft2(ds.img)))
    # normalize it for thresholding in find 2d peaks
    I = I / np.max(I[:])

    # find peaks
    pk_coords = find_peaks2d(arr=I, max_min_neighborhood_size=max_min_fltr_size, threshold=peak_thr)

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
    return dist, I, pk_coords
