import matplotlib.pyplot as plt
import numpy as np
import pdb
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize


def get_histogram_gray(img, bins):
    """Find histogram of grayscale images."""
    m, n = img.shape
    histogram = np.zeros(len(bins)-1, dtype=int)
    for k in range(1, len(bins)):
        s = img[(img < bins[k]).nonzero()]
        t = len((s >= bins[k-1]).nonzero()[0])
        histogram[k-1] += t
    return histogram, np.array(list(bins))


def plot_histograms(histogram, image):
    plt.subplot(121)
    plt.plot(histogram)
    plt.subplot(122)
    plt.imshow(image, 'gray')
    plt.show()


if __name__ == "__main__":
    img = imread("A1_resources/lena.bmp", as_gray=True)
    img = img_as_ubyte(img)

    print("Plotting images at different sizes")
    sizes = [16, 32, 64, 128, 256]
    bins = list(range(0, 257))
    for s in sizes:
        temp = resize(img, (s, s), anti_aliasing=True)
        temp = img_as_ubyte(temp)
        histogram, bins = get_histogram_gray(temp, bins)
        histogram = histogram/histogram.sum()
        pdb.set_trace()
        plot_histograms(histogram, temp)
