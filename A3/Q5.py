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


def sample_img(img, nx, ny):
    m, n = img.shape
    rows = np.array(range(0, m, nx))
    cols = np.array(range(0, n, ny))
    row_sampled = np.vstack(tuple(img[x, :] for x in rows))
    new_img = np.vstack(tuple(row_sampled[:, x].reshape(1, -1) for x in cols))
    return new_img


if __name__ == "__main__":
    img = imread("A3_resources/check.png", as_gray=True)

    for n in range(2, 10):
        img_new = sample_img(img, n, n)
        fft_shifted = np.fft.fftshift(np.fft.fft(img_new))
        mag = scale_img(magnitude(fft_shifted))

        plt.figure(figsize=(10, 10))
        plt.subplots(121)
        plt.imshow(mag)
        plt.title("Image sampled with nx={0}, ny={0}".format(n))
        plt.subplots(122)
        plt.imshow(mag)
        plt.title("FFT of image sampled with nx={0}, ny={0}".format(n))
        plt.show()
