import numpy as np
import pdb

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt


def histogram(img, bins):
    """Get Histogram."""
    histogram, bins = np.histogram(img, bins)
    histogram = histogram/histogram.sum()

    return histogram


def plot_histograms(histogram, image):
    plt.subplot(121)
    plt.plot(histogram)
    plt.subplot(122)
    plt.imshow(image, 'gray')
    plt.show()


if __name__ == "__main__":
    high_contrast = ["A1_resources/hc_1.jpg", "A1_resources/hc_2.jpg", "A1_resources/hc_3.jpeg"]
    low_contrast = ["A1_resources/lc_1.jpg", "A1_resources/lc_2.jpeg", "A1_resources/lc_3.jpg"]

    images = []
    for img_name in high_contrast + low_contrast:
        img = imread(img_name, as_gray=True)
        img = resize(img, (500, 500), anti_aliasing=True)
        img = img_as_ubyte(img)
        images.append(img)

    bins = list(range(0, 257))
    histograms = [histogram(x, bins) for x in images]

    print("High Contrast Histograms:")
    for h, img in zip(histograms[0:3], images[0:3]):
        plot_histograms(h)

    print("Low Contrast Histograms:")
    for h, img in zip(histograms[3::], images[3::]):
        plot_histograms(h)
