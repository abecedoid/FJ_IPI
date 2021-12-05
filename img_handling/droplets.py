import numpy as np
from abc import ABC, abstractmethod


class ParticlePosition(ABC):

    @abstractmethod
    def center(self):
        pass

    @abstractmethod
    def radius(self):
        pass


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

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score: float):
        self._score = score

    def center_coords(self) -> list:
        return [int(self._img.shape[0]/2), int(self._img.shape[1]/2)]

    def rotate_horizontal(self):
        raise NotImplemented


class DropletLabel(ParticlePosition):
    """Holds info about the droplet label
    _points is a list of two points, where first is the center of the circle and the second is some rando
    on the radius

    _shape_type should be circle (so far)"""
    def __init__(self, center_pt: list, radius: int, name: str, shape_type: str = 'circle',
                 fringe_count=None, img_path: str = '', droplet_slice: DropletSlice = None,
                 score=None):
        super(DropletLabel, self).__init__()
        try:
            self.name = name
            self._center = [int(np.round(x)) for x in center_pt]
            self._radius = radius
            self.fringe_count = fringe_count
            self._shape_type = shape_type
            self.img_path = img_path
            self.slice = droplet_slice
            self._score = score
        except Exception as e:
            raise ValueError

    @classmethod
    def init_labelme(cls, labelme_struct: dict, labelme_json_path: str = ''):
        """Overloading constructor to initialize using label me json structure"""
        name = labelme_struct.get('label')
        points = labelme_struct.get('points')

        pt1 = points[0]
        pt2 = points[1]

        # compute radius
        radius = int(np.round(np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)))

        return cls(center_pt=points[0], radius=radius, name=name, img_path=labelme_json_path)

    @classmethod
    def init_dict(cls, json_struct: dict):
        """Overloading constructor to initialize using detector output dict structure"""
        return cls(center_pt=json_struct.get('center'),
                   radius=json_struct.get('radius'),
                   name=json_struct.get('name'),
                   shape_type=json_struct.get('shape'),
                   fringe_count=json_struct.get('fringe_count'),
                   img_path=json_struct.get('img_path'),
                   score=json_struct.get('score'))

    def center(self) -> tuple:
        """returns a tuple of [x, y] denoting droplet's center"""
        # ctr_pt = [int(np.round(x)) for x in self._points[0]]
        return self._center
        # return tuple(ctr_pt)

    def radius(self) -> float:
        """returns radius of the droplet"""
        return self._radius
        # pt1 = self._points[0]
        # pt2 = self._points[1]
        # return int(np.round(np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)))

    def area(self) -> float:
        """returns area of the droplet"""
        return np.pi * self.radius()**2

    def score(self):
        try:
            return self.slice.score
        except Exception as e:
            return self._score

    def __str__(self):
        s = 'DropletLabel:\n'
        s += 'Name: {} \n'.format(self.name)
        s += 'Img path: {} \n'.format(self.img_path)
        s += 'Center: {} \n'.format(self.center())
        s += 'Radius: {} \n'.format(self.radius())
        s += 'Fringe count: {} \n'.format(self.fringe_count)
        s += 'Score: {} \n'.format(self.slice.score)
        return s

    def json(self):
        d = {}
        d['name'] = self.name
        d['center'] = self.center()
        d['radius'] = self.radius()
        d['fringe_count'] = self.fringe_count
        d['shape'] = self._shape_type
        d['img_path'] = self.img_path
        d['score'] = self.score()
        return d


def coords2droplet_labels_list(coords: np.ndarray, circle_radius: int, img_path: str = '') -> list:
    """Takes ndarray of centers of detected circles [N X 2] and converts it to a list of droplet labels"""
    droplets = []

    for i, coord in enumerate(coords):
        name = 'detection_{}'.format(i)
        det_drop = DropletLabel(center_pt=coord, radius=circle_radius, name=name, img_path=img_path)
        droplets.append(det_drop)

    return droplets


if __name__ == '__main__':
    # TEST - loads up image from labelme with the labels and plots them on the image
    PATH = '..//resources//105mm_60deg.6mt18gqf.000099.json'

    from labelme import load_labelme_image, load_labelme_droplet_labels
    from plotters import plot_droplet_list_on_image

    img = load_labelme_image(PATH)
    droplets = load_labelme_droplet_labels(PATH)

    plot_droplet_list_on_image(droplets, img)
