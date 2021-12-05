import sys
sys.path.append('../')
import math
from detector.circle_detector import detect_circles
from img_handling.plotters import *


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


if __name__ == '__main__':

    from PIL import Image

    SHOW_IMAGES = True
    PATH = '../../resources/only_one.png'
    # image = Image.open(PATH)
    img = np.array(Image.open(PATH).convert('L'))
    # todo - use the global detection settings
    coords = detect_circles(img, circle_mask_rad=20, pxcorr1=99.5,
                            pxcorr2=99.5, debug=True)

    det_droplets = []

    # make coords to dropletLabels
    for i, coord in enumerate(coords):
        name = 'detection_{}'.format(i)
        det_drop = DropletLabel(center_pt=coord, radius=40, name=name)
        det_droplets.append(det_drop)

    plot_dict = {'det': det_droplets}
    plot_multiple_droplet_lists_on_image(plot_dict, img)

    print(det_drop)

    # try to extarct the image slice
    print('hehe')

