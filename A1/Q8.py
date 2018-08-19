import matplotlib.pyplot as plt
import numpy as np
import pdb
from skimage import img_as_ubyte
from skimage.io import imread


def BitSlice(img, bit):
    """Find the bit slice from the image."""
    assert bit in list(range(0, 8))
    shape = img.shape
    bit_value = 2**bit
    bit_mask = np.ones(shape, dtype=np.uint8) * bit_value

    bit_slice = np.bitwise_and(img, bit_mask) / bit_value
    return bit_slice


def showSlice(img):
    plt.imshow(img*255, 'gray')
    plt.show()


if __name__ == "__main__":
    # Read image and convert to 8bit grayscale
    img = imread("A1_resources/puppy.jpg", as_gray=True)
    img = img_as_ubyte(img)

    BitSlice(img, 5)
