import numpy as np
from skimage.io import imread
import pdb
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def magnitude(fft_img):
    real = fft_img.real
    imag = fft_img.imag

    mag = np.sqrt(real**2 + imag**2)
    return mag


def filterImg(img, kernel, padding=True):
    img = img.astype(np.float) / 255.0
    m, n = img.shape
    pad_width = int(kernel.shape[0]/2) if padding else 0
    img = np.pad(img, pad_width, 'edge')
    new_img = np.zeros((m, n))

    for i in range(0 + pad_width, m + pad_width):
        for j in range(0 + pad_width, n + pad_width):
            region = img[i-pad_width:i+pad_width, j-pad_width:j+pad_width]
            value = region * kernel
            new_img[i - pad_width, j - pad_width] = value.sum()

    clip_top = (new_img > 1).nonzero()
    new_img[clip_top] = 1

    pdb.set_trace()
# scale = 255/(np.max(new_img) - np.min(new_img))
# new_img = scale*(new_img - np.min(new_img))
    return new_img


def scale_img(img):
    new_img = 20 * np.log(img)
    return new_img


if __name__ == "__main__":
    # f = imread("A3_resources/rice.png", as_gray=True)
    # h = imread("A3_resources/cameraman.jpg", as_gray=True)
    #
    # F = np.fft.fft2(f)
    # F = np.fft.fftshift(F)
    #
    # H = np.fft.fft2(h)
    # H = np.fft.fftshift(H)
    #
    # F_dot_H = F * H
    # idft_F_dot_H = np.fft.ifft2(np.fft.ifftshift(F_dot_H))
    # mag_F_dot_H = magnitude(idft_F_dot_H)
    #
    # f_conv_h = convolve2d(f, h)
    # m, n = f_conv_h.shape
    # central_portion = f_conv_h[m//4:m//4 + m//2 + 1, m//4:m//4 + m//2 + 1]
    #
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(mag_F_dot_H, 'gray')
    # plt.title('iDFT(FH)')
    #
    # plt.subplot(122)
    # plt.imshow(central_portion, 'gray')
    # plt.title('Central portion of f*h')
    # plt.show()
    #
    # dist = np.mean((mag_F_dot_H - central_portion) ** 2)
    # print("Average Squared distance between the two images is: {}".format(dist))

    f = imread("A3_resources/5.jpeg", as_gray=True)

    h = imread("A3_resources/64x64.png", as_gray=True)
    h_pad = np.pad(h, ((143, 143), (166, 167)), 'constant', constant_values=(0))

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(f, 'gray')
    plt.title('f of size 350x397')

    plt.subplot(122)
    plt.imshow(h, 'gray')
    plt.title('h of size 64x64 with zero padding')
    plt.show()

    F = np.fft.fft2(f)
    H = np.fft.fft2(h_pad)
    FH = np.fft.fftshift(F*H)
    FH = scale_img(magnitude(FH))

    f_conv_h = convolve2d(f, h)
    f_conv_h = f_conv_h[32:-31, 32:-31]

    

    pdb.set_trace()
