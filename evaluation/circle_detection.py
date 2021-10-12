import numpy as np
import math
from detector.circle_detector import detect_circles_labelme, detect_circles
from helpers.labeled_jsons import *
import matplotlib.pyplot as plt


def two_circle_area_intersection(c1: DropletLabel, c2: DropletLabel) -> float:
    """Finds intersection area of two circles.
    Returns intersection area of two circles otherwise 0
    """

    d = math.dist(c1.center(), c2.center())
    rad1sqr = c1.radius() ** 2
    rad2sqr = c2.radius() ** 2

    if d == 0:
        # the circle centers are the same
        return math.pi * min(c1.radius(), c2.radius()) ** 2

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1.radius() * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2.radius() * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = math.acos(angle1) * 2
        theta2 = math.acos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * math.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * math.sin(theta1))

        return area1 + area2
    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return math.pi * min(c1.radius(), c2.radius()) ** 2
    return 0


def circ_intersection_over_union(truth: DropletLabel, det: DropletLabel):
    """Computes the intersection over union for two DropletLabels"""
    # compute intersection
    intersection = two_circle_area_intersection(truth, det)
    # compute union
    union = truth.area() + det.area()
    # intersection over union
    iou = intersection / union
    return iou

def find_closest_point(point: np.ndarray, points: np.ndarray):
    """"""
    pass

# def closest_node(node, nodes):
#     nodes = np.asarray(nodes)
#     dist_2 = np.sum((nodes - node)**2, axis=1)
#     return np.argmin(dist_2)


if __name__ == '__main__':


    PATH = '..//resources//105mm_60deg.6mt18gqf.000099.json'
    img = load_labelme_image(PATH)
    gt_droplets = load_labelme_droplet_labels(PATH)
    coords = detect_circles(img)

    ious = []

    for gt_drop in gt_droplets:

        # make coords to dropletLabels
        for i, coord in enumerate(coords):
            name = 'detection_{}'.format(i)
            det_drop = DropletLabel(center_pt=coord, radius=52, name=name)

            ious.append(circ_intersection_over_union(truth=gt_drop, det=det_drop))


    plt.figure()
    plt.plot(ious)
    plt.show()
    print('sum of ious: {}'.format(sum(ious)))

