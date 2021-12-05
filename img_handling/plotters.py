import numpy as np
import cv2
from img_handling.droplets import DropletLabel


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

    for droplet in droplet_list:
        img = cv2.circle(img, center=droplet.center(), radius=droplet.radius(), color=(0, 0, 255), thickness=2)

    cv2.imshow('droplet', img)
    cv2.waitKey(0)


def plot_multiple_droplet_lists_on_image(dict_of_lists: dict, img: np.ndarray, wait_key=True):
    # maximum 3 sets of droplets can be plotted so far...
    assert len(dict_of_lists.keys()) <= 3
    # ensure image is in rgb so that colors are existing
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    # todo - move to external config
    font_settings = {'bottom_left_corner': (40, 40),
                     'font': cv2.FONT_HERSHEY_SIMPLEX,
                     'font_scale': 1,
                     'font_color': (255, 255, 255),
                     'thickness': 1,
                     'line_type': 2}

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

    if wait_key:
        cv2.waitKey(0)


def plot_points_on_image(point_list: list, img: np.ndarray):
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for pt in point_list:
        img = cv2.circle(img, center=(int(pt[-1]), int(pt[0])), radius=0, color=(0, 0, 255), thickness=10)

    cv2.imshow('droplet', img)
    cv2.waitKey()
