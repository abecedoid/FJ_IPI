import numpy as np
from helpers.labeled_jsons import DropletLabel


class DropletSlice(object):
    def __init__(self, img: np.ndarray, radius: int, score=None):
        self._img = img
        self._radius = radius
        self._score = score

    @property
    def img(self):
        return self._img

    @property
    def radius(self):
        return self._radius


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

    img_slice = img[lx: lx + side, ly: ly + side]

    return DropletSlice(img_slice, droplet_label.radius(), score=None)


def count_fringes(ds: DropletSlice):
    """Takes the image and estimates number of fringes in it"""
    raise NotImplementedError