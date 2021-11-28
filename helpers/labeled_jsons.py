import json
import numpy as np
# from image_utils import img_b64_to_arr
from helpers import image_utils
import os
import cv2
from helpers.interfaces import ParticlePosition


def load_labelme_image(path2json: str) -> np.ndarray:
    """Loads image from string data contained in the LabelMe's JSON"""
    with open(path2json, 'r') as f:
        data = json.load(f)

    image_data = data.get('imageData')

    img = image_utils.img_b64_to_arr(image_data)

    return img


def load_labelme_droplet_labels(path2json: str) -> list:
    """Loads a list of labelme's droplet labels"""
    with open(path2json, 'r') as f:
        data = json.load(f)

    shapes_data = data.get('shapes')

    droplets = []
    for shape in shapes_data:
        try:
            drop = DropletLabel.init_labelme(labelme_struct=shape, labelme_json_path=path2json)
            droplets.append(drop)
            # droplets.append(DropletLabel.init_labelme(labelme_struct=shape))
        except Exception as e:
            droplets.append(DropletLabel(center_pt=[0, 0], radius=0, name='fail'))
            print('Failed to load DropletLabel from {} because {}'.format(shape, e))

    return droplets


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


def plot_image(img: np.ndarray) -> None:
    """plots image"""
    cv2.namedWindow('droplets', cv2.WINDOW_NORMAL)
    cv2.imshow('droplets', img)
    cv2.waitKey()


def plot_droplet_on_image(droplet: DropletLabel, img: np.ndarray):
    """plots single droplet on a given image"""
    # ensure image is in rgb so that colors are existing
    assert len(img.shape) == 2 or len(img.shape) == 3

    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_circle = cv2.circle(img, center=droplet.center(), radius=droplet.radius(), color=(0, 0, 255), thickness=2)
    cv2.imshow('droplet', img_circle)
    cv2.waitKey(0)


def plot_droplet_list_on_image(droplet_list: list, img: np.ndarray):
    # ensure image is in rgb so that colors are existing
    assert len(img.shape) == 2 or len(img.shape) == 3

    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for droplet in droplets:
        img = cv2.circle(img, center=droplet.center(), radius=droplet.radius(), color=(0, 0, 255), thickness=2)

    cv2.imshow('droplet', img)
    cv2.waitKey(0)


def plot_multiple_droplet_lists_on_image(dict_of_lists: dict, img: np.ndarray):
    # maximum 3 sets of droplets can be plotted so far...
    assert len(dict_of_lists.keys()) <= 3
    # ensure image is in rgb so that colors are existing
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    # todo - move to external config
    font_settings = {}
    font_settings['bottom_left_corner'] = (40, 40)
    font_settings['font'] = cv2.FONT_HERSHEY_SIMPLEX
    font_settings['font_scale'] = 1
    font_settings['font_color'] = (255, 255, 255)
    font_settings['thickness'] = 1
    font_settings['line_type'] = 2

    legend_text = ''

    for i, key in enumerate(dict_of_lists.keys()):
        droplets = dict_of_lists[key]
        legend_text += str(key) + ': ' + str(COLORS[i]) + ' (BGR) || '
        for droplet in droplets:
            try:
                img = cv2.circle(img, center=droplet.center(), radius=droplet.radius(), color=COLORS[i], thickness=2)
            except Exception as e:
                print('failed to draw circle, because {}'.format(e))

    cv2.namedWindow('droplets', cv2.WINDOW_NORMAL)
    cv2.putText(img, legend_text,
                font_settings['bottom_left_corner'],
                font_settings['font'],
                font_settings['font_scale'],
                font_settings['font_color'],
                font_settings['thickness'],
                font_settings['line_type'])
    cv2.imshow('droplets', img)
    cv2.waitKey(0)


def plot_points_on_image(point_list: list, img: np.ndarray):
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for pt in point_list:
        img = cv2.circle(img, center=(int(pt[-1]), int(pt[0])), radius=0, color=(0, 0, 255), thickness=10)

    cv2.imshow('droplet', img)
    cv2.waitKey()


def coords2droplet_labels_list(coords: np.ndarray, circle_radius: int, img_path: str = '') -> list:
    """Takes ndarray of centers of detected circles [N X 2] and converts it to a list of droplet labels"""
    droplets = []

    for i, coord in enumerate(coords):
        name = 'detection_{}'.format(i)
        det_drop = DropletLabel(center_pt=coord, radius=circle_radius, name=name, img_path=img_path)
        droplets.append(det_drop)

    return droplets



if __name__ == '__main__':
    # TEST
    PATH = '..//resources//105mm_60deg.6mt18gqf.000099.json'

    img = load_labelme_image(PATH)
    droplets = load_labelme_droplet_labels(PATH)

    plot_droplet_list_on_image(droplets, img)
    # for d in droplets:
    #     print(d)
    #
    #     plot_droplet_on_image(droplet=d, img=img)

    # img = load_labelme_image(PATH)
    # import PIL
    # import PIL.ImageDraw
    #
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # # image_rgb = cv2.circle(img, (295, 49), radius=0, color=(255, 0, 0), thickness=10)
    # image_rgb = cv2.circle(img, (318, 70), radius=0, color=(255, 0, 0), thickness=10)
    # cv2.imshow('name', image_rgb)
    # cv2.waitKey()

    # image = PIL.Image.fromarray(img)
    # draw = PIL.ImageDraw.Draw(image)
    # draw.ellipse((295, 295, 49, 49), outline='red', fill=128)
    # image.show()


# print('hehe')

