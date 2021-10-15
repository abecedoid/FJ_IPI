import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import scipy.ndimage as nd
import skimage.feature as ftr
from helpers.labeled_jsons import load_labelme_image, plot_points_on_image
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


def gaussian_kernel(l=5, sig=3):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def find_peaks2d(arr: np.ndarray, max_min_neighborhood_size: int, threshold: float) -> np.ndarray:
    # todo - rewrite the loop so that its faster
    """Applies maximum filter for given area, then creates bool array with true values, where
    the maxima lay in the original image.
    Applies minimum filter to the original image and creates a difference image between max and min image
    Detected peaks are located where the difference between the min and max image is smaller than threshold

    returns ndarray [N x 2] containing the x, y coords of the peaks"""
    data_max = nd.maximum_filter(arr, max_min_neighborhood_size)
    maxima = (arr == data_max)
    data_min = nd.minimum_filter(arr, max_min_neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = nd.label(maxima)
    slices = nd.find_objects(labeled)
    coords = np.zeros((len(slices), 2))
    for i, (dy, dx) in enumerate(slices):
        x_center = (dx.start + dx.stop - 1) / 2
        coords[i, 0] = x_center
        y_center = (dy.start + dy.stop - 1) / 2
        coords[i, 1] = y_center

    return coords


def detect_circles_labelme(img_path: str, DS_COEFF: int=4) -> (np.ndarray, np.ndarray):
    """Just a convenience wrapper for the labelme format jsons"""
    img = load_labelme_image(img_path)
    coords = detect_circles(img, DS_COEFF)
    return img, coords


def detect_circles(img: np.ndarray, DS_COEFF: int = 2,
                   circle_mask_rad: int = 15, circle_mask_wdth: int = None, circle_mask_radoff_size: int = 5,
                   pxcorr1: int = 90, pxcorr2: int = 95, peak_min_dist: int = 7, debug=False) -> np.ndarray:
    """Returns ndarray of centers of detected circles [N X 2]"""
    # downsample
    print('original img res is: {}'.format(img.shape))
    new_res = [round(x / DS_COEFF) for x in img.shape]
    new_res.reverse()  # cv2 is [height, width] dims!

    img_ds = cv2.resize(img, new_res, interpolation=cv2.INTER_AREA)
    print('resized image size is {}'.format(new_res))

    # get circle mask for xcorr
    side_size = 2 * circle_mask_rad + circle_mask_radoff_size
    mask = create_circular_mask(h=side_size, w=side_size, radius=circle_mask_rad, circle_width=circle_mask_wdth)

    # first correlation
    corr = normxcorr2(mask, img_ds, mode='same')
    p95 = np.percentile(corr[:], pxcorr1)
    corr[corr < p95] = 0

    # second correlation
    corr2 = signal.correlate2d(corr, gaussian_kernel(), mode='same')
    p95 = np.percentile(corr2[:], pxcorr2)
    corr2[corr2 < p95] = 0

    coords = find_peaks2d(arr=corr2, max_min_neighborhood_size=int(circle_mask_rad), threshold=0.00001)

    if debug:
        plt.figure()
        plt.subplot(221)
        plt.imshow(corr)
        plt.subplot(222)
        plt.imshow(corr2)
        plt.scatter(coords[:, 0], coords[:, 1], color='r', s=1)
        plt.subplot(223)
        plt.imshow(img_ds)
        plt.scatter(coords[:, 0], coords[:, 1], color='r', s=1)
        plt.subplot(224)
        plt.imshow(corr2)
        plt.scatter(coords[:, 0], coords[:, 1], color='r', s=1)

        plt.show()

    # de-downsample coords (ndarray (N, 2))
    coords = coords * DS_COEFF

    return coords


if __name__ == '__main__':
    # demicko
    path = '../resources/105mm_60deg.6mt18gqf.000099.json'
    path = os.path.abspath(os.path.join(os.getcwd(), path))

    img, coords = detect_circles_labelme(img_path=path)

    print(coords)

    plot_points_on_image(coords, img)


