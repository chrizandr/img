import numpy as np
from skimage.io import imread
import pdb
import matplotlib.pyplot as plt


def magnitude(fft_img):
    real = fft_img.real
    imag = fft_img.imag

    mag = np.sqrt(real**2 + imag**2)
    return mag


def double_image(img):
    rows, cols = img.shape
    col_pad = np.zeros((rows, cols))
    padded_img = np.hstack((col_pad, img))

    rows, cols = padded_img.shape
    row_pad = np.zeros((rows, cols))
    padded_img = np.vstack((padded_img, row_pad))

    return padded_img


def scale_img(img):
    new_img = 20 * np.log(img)
    return new_img


if __name__ == "__main__":
    img = imread("A3_resources/64x64.png", as_gray=True)
    new_img = img

    for i, n in [64, 128, 256, 512]:
        plt.imshow(new_img, 'gray')
        plt.title("Image of size {0}x{0}".format(n))
        plt.show()
        fft = np.fft.fft2(new_img)
        fft_shifted = np.fft.fftshift(fft)
        mag = magnitude(fft_shifted)
        scale_mag = scale_img(mag)
        plt.imshow(scale_mag, 'gray')
        plt.title("Fourier transform for {0}x{0} image".format(n))
        plt.show()

        new_img = double_image(new_img)
