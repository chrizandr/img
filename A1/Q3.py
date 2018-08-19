import numpy as np
import pdb
from skimage import img_as_float
from skimage.io import imread

import matplotlib.pyplot as plt


def transformIntensity(img, K1, K2, a, b):
    """Linear intensity transform."""
    valid_pixels = np.where(np.logical_and(img >= a, img <= b))

    new_img = img.copy()
    new_img[valid_pixels] = K1 * img[valid_pixels] + K2
    pdb.set_trace()
    return new_img


def showImages(image, result):
    plt.subplot(121)
    plt.imshow(image, 'gray')
    plt.subplot(122)
    plt.imshow(result, 'gray')
    plt.show()


if __name__ == "__main__":
    # Read image and convert to 8bit grayscale
    img = imread("A1_resources/lena.bmp", as_gray=True)
    img = img_as_float(img)
    transformed = transformIntensity(img, 1, 0.25, 0, 0.75)
    transformed = transformIntensity(transformed, 0, 0, 0.75, 1)
    pdb.set_trace()
    showImages(img, transformed)

    # Do histogram equilization
