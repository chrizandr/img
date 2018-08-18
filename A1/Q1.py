"""Functions and definitions for Q1."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pdb
from skimage import img_as_ubyte
from skimage.io import imread


def hexencode(rgb):
    """Encode colors to matplotlib spec."""
    if type(rgb) is tuple:
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
    else:
        r = rgb
        g = rgb
        b = rgb
        return '#%02x%02x%02x' % (r, g, b)


def show_colors(img, colors, gray=False):
    """Plot colormap."""
    colors = [hexencode(x) for x in colors]
    dom_color = colors[0]
    # plt

    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(311)
    if gray:
        ax.imshow(img, 'gray')
    else:
        ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Image")

    ax = fig.add_subplot(312)
    ax.add_patch(
        patches.Rectangle(
            (0, 0), 1, 1, facecolor=dom_color
        )
    )
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Most dominant color")

    ax = fig.add_subplot(313)
    for x, color in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x, 0), 1, 1, facecolor=color
            )
        )
    ax.set_xlim((0, len(colors)))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Dominant colors")

    return fig


def get_histogram_gray(img, bins):
    """Find histogram of grayscale images."""
    m, n = img.shape
    histogram = np.zeros(len(bins)-1, dtype=int)
    for k in range(1, len(bins)):
        s = img[(img < bins[k]).nonzero()]
        t = len((s >= bins[k-1]).nonzero()[0])
        histogram[k-1] += t
    return histogram, np.array(list(bins))


def get_histogram_color(img):
    """Find histogram of color images."""
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            


def domColors_gray(img, k):
    """Find k most dominant colors in grayscale image."""
    # Find histogram of grayscale colors, with bin size as 5 to avoid same color
    histogram, bins = get_histogram_gray(img, bins=range(0, 257, 5))
    # Sort histogram to find dominant colors
    colors = np.flip(np.argsort(histogram))
    # Place colors in bins of size 5, to merge similar colors as one
    k_colors = bins[colors[0:k]]
    return k_colors


def domColors_color(img, k):
    """Find k most dominant colors in grayscale image."""



def domColors(img, k):
    """Find k most dominant colors in image."""
    if len(img.shape) == 2:
        return domColors_gray(img, k)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        return domColors_color(img, k)
    else:
        raise ValueError("Invalid image")


if __name__ == "__main__":
    img = imread("test.jpg", as_gray=True)
    grayimg = img_as_ubyte(img)
    colors = domColors(grayimg, 10)
    show_colors(grayimg, colors, True).show()
    pdb.set_trace()
