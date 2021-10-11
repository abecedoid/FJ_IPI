import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import scipy.ndimage as nd
import skimage.feature as ftr


def create_circular_mask(h, w, center=None, radius=None, circle_width=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if circle_width is None:
        mask = dist_from_center <= radius
    else:
        mask_large = dist_from_center <= radius
        mask_small = dist_from_center >= (radius - circle_width)
        mask = np.logical_and(mask_small, mask_large)

    mask = mask.astype('uint8') * 255
    return mask


def resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


# mask = create_circular_mask(h = 100, w = 100, radius=35, circle_width=3)
# plt.imshow(mask)
# plt.show()

from helpers.labeled_jsons import load_labelme_image
#
path = '../resources/only_one.png'
path = os.path.abspath(os.path.join(os.getcwd(), path))
print('path is: {}'.format(path))
# img = load_labelme_image(path2json=path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img[img < 20] = 0
# img = cv2.equalizeHist(img)
# img = cv2.convertScaleAbs(img, alpha=1.3, beta=40)

cv2.imshow('hh', img)
cv2.waitKey(0)

# DS image
# img = resize_image(img, (256, 320))
# img = cv2.resize(img, (256, 320))
plt.imshow(img)
plt.show()

# do the fft2
fftimg = np.fft.fftshift(np.fft.fft2(img))
plt.imshow(np.log(abs(fftimg)))
plt.show()

print('image resolution is {} x {}'.format(img.shape[0], img.shape[1]))

# get mask
mask = create_circular_mask(h=100, w=100, radius=40, circle_width=1)

plt.imshow(mask)
plt.show()

fftmask = np.fft.fftshift(np.fft.fft2(mask))
plt.imshow(np.log(abs(fftmask)))
plt.show()

fftimg_a = abs(fftimg)
fftmask_a = abs(fftmask)

corr = signal.correlate2d(fftimg_a.astype(np.float64), fftmask_a.astype(np.float64))
corr = corr - np.mean(corr[:])
corr = corr ** 2
corr = corr / np.amax(corr)

# corr = signal.correlate2d(img.astype(np.float64), mask.astype(np.float64))
# corr = corr - np.mean(corr[:])
# corr = corr ** 2
# corr = corr / np.amax(corr)
# corr[corr < 1] = 0
print('debug')

# coords = ftr.peak_local_max(corr, min_distance=5)
# from helpers.labeled_jsons import plot_points_on_image
# img = plot_points_on_image(coords, img)

# for i, coord in enumerate(coords):
#     coords[i, :] = (coord * ds_ratio).astype(dtype=np.uint16)

# for pt in coords:
#     img = cv2.circle(img, center=(pt[0], pt[1]), radius=0, color=(0, 0, 255), thickness=10)

# cv2.imshow('origo image with markers', img.astype('uint8'))
# cv2.waitKey()

# plt.imshow(img)
# plt.show()
plt.imshow(corr)
plt.show()


