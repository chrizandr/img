import numpy as np
import pdb

from skimage import img_as_ubyte
from skimage.io import imread
import matplotlib.pyplot as plt


def HistogramEq(img):
    """Does Histogram Equilization."""
    histogram, bins = np.histogram(img, bins=list(range(0, 257)))
    histogram = histogram/histogram.sum()

    cdf = histogram.cumsum()
    transformation = np.uint8(255 * cdf)
    m, n = img.shape
    Y = np.zeros_like(img)

    for i in range(0, 256):
        pixels = (img == i).nonzero()
        Y[pixels] = transformation[i]

    return Y


def local_HistogramEq(img, k):
    """Local Histogram Equilization."""
    m, n = img.shape
    new_img = img.copy()
    for i in range(k//2, m-(k//2)):
        for j in range(k//2, n-(k//2)):
            window = img[i-k//2: i+k//2, j-k//2:j+k//2]
            equilized_img = HistogramEq(window)
            new_img[i, j] = equilized_img[k//2, k//2]
    return new_img


def showImages(img, result):
    """Show the two images."""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(211)
    ax.imshow(img, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Original")

    ax = fig.add_subplot(212)
    ax.imshow(result, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Equalized")

    plt.show()


def showImages2(src, trgt, matched):
    """Show the two images."""
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(src, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Source")

    ax = fig.add_subplot(132)
    ax.imshow(trgt, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Target")

    ax = fig.add_subplot(133)
    ax.imshow(matched, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Matched")

    plt.show()


def hist_match(source, target):
    """Histogram Matching."""
    shape = target.shape
    target = target.ravel()
    source = source.ravel()

    pixel_values, bins, counts = np.unique(target, return_inverse=True,
                                           return_counts=True)
    target_pixels, target_counts = np.unique(source, return_counts=True)

    s_cdf = np.cumsum(counts).astype(np.float64)
    t_cdf = np.cumsum(target_counts).astype(np.float64)

    s_cdf = s_cdf/s_cdf[-1]
    t_cdf = t_cdf/t_cdf[-1]

    new_values = np.interp(s_cdf, t_cdf, target_pixels)
    new_image = new_values[bins]
    new_image = new_image.reshape(shape)

    return new_values[bins].reshape(shape)


if __name__ == "__main__":
    # Read image and convert to 8bit grayscale
    img = imread("A1_resources/hist_equal2.jpg", as_gray=True)
    img = img_as_ubyte(img)

    # Do histogram equilization
    result = local_HistogramEq(img, 7)

    # See the results
    showImages(img, result)
    pdb.set_trace()
