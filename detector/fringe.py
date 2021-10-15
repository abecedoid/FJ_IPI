import numpy as np


class DropletSlice():
    def __init__(self):
        self._img
        self._radius
        self._score
        # todo


def get_droplet_slices(img, coords: np.ndarray) -> list:
    """Returns a list of droplet slices"""
    pass


def count_fringes(ds: DropletSlice):
    """Takes the image and estimates number of fringes in it"""
    raise NotImplementedError