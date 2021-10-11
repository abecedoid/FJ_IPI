import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import scipy.ndimage as nd
import skimage.feature as ftr

import numpy as np
from scipy.signal import fftconvolve


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


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
path = '../resources/105mm_60deg.6mt18gqf.000099.json'
path = os.path.abspath(os.path.join(os.getcwd(), path))
print('path is: {}'.format(path))
img = load_labelme_image(path2json=path)

# img[img < 20] = 0
# img = cv2.equalizeHist(img)
# img = cv2.convertScaleAbs(img, alpha=1.3, beta=40)

# cv2.imshow('hh', img)
# cv2.waitKey(0)

# DS image
img = resize_image(img, (256, 320))
img_og = img.copy()
# img = cv2.resize(img, (256, 320))
# plt.imshow(img)
# plt.show()

print('image resolution is {} x {}'.format(img.shape[0], img.shape[1]))

# get mask
mask = create_circular_mask(h=20, w=20, radius=7, circle_width=1)

# plt.imshow(mask)
# plt.show()

# mask = abs(np.fft.fftshift(np.fft.fft2(mask)))

# corr = signal.correlate2d(img.astype(np.float64), mask.astype(np.float64))
# corr = corr - np.mean(corr[:])
# corr = corr ** 2
# corr = corr / np.amax(corr)
# corr = corr * 255
# corr[corr < 1] = 0
print('debug')

corr = normxcorr2(mask, img)
p95 = np.percentile(corr[:], 99)
corr[corr < p95] = 0
# corr = corr * 255

# corr = corr.astype(np.uint8)
# corr = cv2.medianBlur(corr, 1)

# correlate with gaussian
def gkern(l=5, sig=3):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

corr2 = signal.correlate2d(corr, gkern())
p95 = np.percentile(corr2[:], 99)
corr2[corr2 < p95] = 0
corr2 = corr2[11:-11, 11:-11]

coords = ftr.peak_local_max(corr2, min_distance=10)
# coords = np.fliplr(coords)
from helpers.labeled_jsons import plot_points_on_image
img = plot_points_on_image(coords, img)

# for i, coord in enumerate(coords):
#     coords[i, :] = (coord * ds_ratio).astype(dtype=np.uint16)

# for pt in coords:
#     img = cv2.circle(img, center=(pt[0], pt[1]), radius=0, color=(0, 0, 255), thickness=10)

# cv2.imshow('origo image with markers', img.astype('uint8'))
# cv2.waitKey()


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
# surf = ax.plot3D(np.linspace(np.linspace(0, 256), np.linspace(0, 320), corr))
# plt.show()
plt.figure()
plt.subplot(131)
plt.imshow(corr)
plt.subplot(132)
plt.imshow(img_og)
plt.subplot(133)
plt.imshow(corr2)
plt.show()


