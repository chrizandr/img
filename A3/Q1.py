import numpy as np
from skimage.io import imread
import pdb
import matplotlib.pyplot as plt


def magnitude(fft_img):
    real = fft_img.real
    imag = fft_img.imag

    mag = np.sqrt(real**2 + imag**2)
    return mag


def scale_img(img):
    new_img = 20 * np.log(img)
    return new_img


if __name__ == "__main__":
    img = imread("A3_resources/5.jpeg", as_gray=True)
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    mag = magnitude(fft_shifted)
    scale_mag = scale_img(mag)
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title("Original Image")
    plt.subplot(223)
    plt.imshow(scale_mag, 'gray')
    plt.title("Original FFT magnitude")

    # Since we take a log of the intensiry, the change in intensity is low
    # High frequency components are removed from the image and thus there is less high frequency in the FFT

    new_img = scale_img(img.astype(np.float))
    new_fft = np.fft.fft2(new_img)
    new_fft_shifted = np.fft.fftshift(new_fft)
    new_mag = magnitude(new_fft_shifted)
    new_scale_mag = scale_img(new_mag)
    plt.subplot(222)
    plt.imshow(new_img, 'gray')
    plt.title("Scaled Image")
    plt.subplot(224)
    plt.imshow(new_scale_mag, 'gray')
    plt.title("Scaled Image FFT magnitude")
    plt.show()

    # FFT of FFT
    fft_fft = np.fft.fft2(fft)
    mag = magnitude(fft_fft)

    plt.imshow(mag, 'gray')
    plt.title("FFT of the FFT")
    plt.show()

    fft_flip = np.fft.ifftshift(np.flipud(fft_shifted))
    fft_fft_flip = np.fft.fft2(fft_flip)
    mag = magnitude(fft_fft_flip)

    plt.imshow(mag, 'gray')
    plt.title("FFT of the FFT after flipping")
    plt.show()

    pdb.set_trace()
