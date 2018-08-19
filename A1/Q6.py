import numpy as np
import pdb

from skimage import img_as_ubyte
from skimage.io import imread
import matplotlib.pyplot as plt


def BitQuantized(img, k):
    """Bit Quantized image."""
    quant = 2**k - 1
    if quant > 255 or quant < 1:
        raise ValueError("Invalid value for k")

    m, n = img.shape

    scaled_img = img.astype(np.float)
    scaled_img = scaled_img/255

    quant_img = (scaled_img * quant).astype(np.int)

    rescaled_img = quant_img.astype(np.float)
    rescaled_img = rescaled_img/quant

    new_img = (rescaled_img * 255).astype(np.uint8)
    return new_img


def showImages(img, result):
    """Show the two images."""
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Original")

    ax = fig.add_subplot(122)
    ax.imshow(result, 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Quantized")

    plt.show()


def plot_Quantized(img):
    """Plot BitQuantized images for all valid k."""
    for k in range(1, 8):
        new_img = BitQuantized(img, k)
        print("{} bit Quantized image".format(k))
        showImages(img, new_img)


if __name__ == "__main__":
    # Read image and convert to 8bit grayscale
    img = imread("A1_resources/puppy.jpg", as_gray=True)
    img = img_as_ubyte(img)

    # Do histogram equilization
    result = BitQuantized(img, 1)

    # See the results
    showImages(img, result)
    pdb.set_trace()
