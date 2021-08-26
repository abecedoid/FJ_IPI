import json
import numpy as np
# from image_utils import img_b64_to_arr
import image_utils
import os
import cv2


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
            droplets.append(DropletLabel(label_struct=shape))
        except Exception as e:
            print('Failed to load DropletLabel from {} because {}'.format(shape, e))

    return droplets


class DropletLabel(object):
    """Holds info about the droplet label
    _points is a list of two points, where first is the center of the circle and the second is some rando
    on the radius

    _shape_type should be circle (so far)"""
    def __init__(self, label_struct: dict):
        try:
            self._name = label_struct.get('label')
            self._points = label_struct.get('points')
            if len(self._points) != 2:
                raise ValueError
            self._shape_type = label_struct.get('shape_type')
        except Exception as e:
            raise ValueError

    def center(self) -> tuple:
        """returns a list of [x, y] denoting droplet's center"""
        ctr_pt = [int(np.round(x)) for x in self._points[0]]
        return tuple(ctr_pt)

    def radius(self) -> float:
        """returns radius of the droplet"""
        pt1 = self._points[0]
        pt2 = self._points[1]
        return int(np.round(np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)))

    def __str__(self):
        s = 'DropletLabel:\n'
        s += 'Name: {} \n'.format(self._name)
        s += 'Center: {} \n'.format(self.center())
        s += 'Radius: {} \n'.format(self.radius())
        return s


def plot_droplet_on_image(droplet: DropletLabel, img: np.ndarray):
    """plots single droplet on a given image"""
    # ensure image is in rgb so that colors are existing
    assert len(img.shape) == 2 or len(img.shape) == 3

    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_circle = cv2.circle(img, center=droplet.center(), radius=droplet.radius(), color=(0, 0, 255), thickness=2)
    cv2.imshow('droplet', img_circle)
    cv2.waitKey(0)




if __name__ == '__main__':
    # TEST
    PATH = '..//resources//105mm_60deg.6mt18gqf.000099.json'

    img = load_labelme_image(PATH)
    droplets = load_labelme_droplet_labels(PATH)

    for d in droplets:
        print(d)

        plot_droplet_on_image(droplet=d, img=img)

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

