import numpy as np
import pdb

from skimage import img_as_ubyte
from skimage.io import imread

from skimage.filters import threshold_otsu, threshold_local

import matplotlib.pyplot as plt


def BinarizeImage(image, method="global", block_size=35, offset=10):
    if method == "global":
        thresh = threshold_otsu(image)
        binary = np.where(image > thresh, 1, 0)
        return binary
    elif method == "adaptive":
        thresh = threshold_local(image, block_size=block_size, offset=offset)
        binary = np.where(image > thresh, 1, 0)
        return binary


def showImages(image, result):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(121)
    ax.imshow(image, 'gray')
    ax = fig.add_subplot(122)
    ax.imshow(result, 'gray')
    plt.show()


if __name__ == "__main__":
    # Read image and convert to 8bit grayscale
    img = imread("A1_resources/palm-leaf-2.jpg", as_gray=True)
    img = img_as_ubyte(img)
    binary = BinarizeImage(img, "adaptive")
    showImages(img, binary)
