import numpy as np
from skimage.io import imread
import pdb
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import resize
import time


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

    # ---------------------------------------------
    #
    # f = imread("A3_resources/5.jpeg", as_gray=True)
    #
    # h = imread("A3_resources/64x64.png", as_gray=True)
    # h_pad = np.pad(h, ((143, 143), (166, 167)), 'constant', constant_values=(0))
    #
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(f, 'gray')
    # plt.title('f of size 350x397')
    #
    # plt.subplot(122)
    # plt.imshow(h, 'gray')
    # plt.title('h of size 64x64 with zero padding')
    # plt.show()
    #
    # F = np.fft.fft2(f)
    # H = np.fft.fft2(h_pad)
    # FH = np.fft.fftshift(F*H)
    # FH = scale_img(magnitude(FH))
    #
    # f_conv_h = convolve2d(f, h)
    #
    # DFT_fh = np.fft.fft2(f_conv_h)
    # DFT_fh = np.fft.fftshift(DFT_fh)
    # DFT_fh = scale_img(magnitude(DFT_fh))
    #
    # plt.figure(figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(FH, 'gray')
    # plt.title('Multiplication of Fourier transformed images')
    #
    # plt.subplot(122)
    # plt.imshow(DFT_fh, 'gray')
    # plt.title('DFT of convolved images.')
    # plt.show()

    # ---------------------------------------------
    #
    # f = imread("A3_resources/1.tiff", as_gray=True)
    # h_original = imread("A3_resources/2.tiff", as_gray=True)
    #
    # freq_time = []
    # spat_time = []
    # for n in [32, 64, 128, 256]:
    #     h = resize(h_original, (n, n))
    #     row_pad = (f.shape[0] - h.shape[0]) // 2
    #     col_pad = (f.shape[1] - h.shape[1]) // 2
    #     start = time.time()
    #     h_pad = np.pad(h, ((row_pad, row_pad), (col_pad, col_pad)), 'constant', constant_values=(0))
    #     F = np.fft.fft2(f)
    #     H = np.fft.fft2(h_pad)
    #     iDFT_FH = np.fft.ifft(F*H)
    #     iDFT_FH = scale_img(magnitude(iDFT_FH))
    #     end = time.time()
    #     freq_time.append(float(str(end-start)))
    #
    #     start = time.time()
    #     f_conv_h = convolve2d(f, h)
    #     end = time.time()
    #     spat_time.append(float(str(end-start)))
    #
    # plt.plot(freq_time, [32, 64, 128, 256], color='red', label="Time taken in Frequency Domain")
    # plt.plot(spat_time, [32, 64, 128, 256], color='blue', label="Time taken in Spatial Domain")
    #
    # plt.legend()
    # plt.xlabel("Image Size")
    # plt.ylabel("Time")
    # plt.show()

    # ---------------------------------------------

    f = imread("A3_resources/cameraman.jpg", as_gray=True)
    h = imread("A3_resources/rice.png", as_gray=True)

    start = time.time()
    f_pad = np.pad(f, ((128, 127), (128, 127)), 'constant', constant_values=(0))
    h_pad = np.pad(h, ((128, 127), (128, 127)), 'constant', constant_values=(0))

    F = np.fft.fftshift(np.fft.fft2(f_pad))
    H = np.fft.fftshift(np.fft.fft2(h_pad))
    f_conv_h = np.fft.ifft(np.fft.ifftshift(F * H))
    f_conv_h_1 = scale_img(magnitude(f_conv_h))
    end = time.time()

    print("Time requried to compute = {}s".format(end-start))

    start = time.time()
    f_pad = np.pad(f, ((128, 128), (128, 128)), 'constant', constant_values=(0))
    h_pad = np.pad(h, ((128, 128), (128, 128)), 'constant', constant_values=(0))

    F = np.fft.fftshift(np.fft.fft2(f_pad))
    H = np.fft.fftshift(np.fft.fft2(h_pad))
    f_conv_h = np.fft.ifft(np.fft.ifftshift(F * H))
    f_conv_h_2 = scale_img(magnitude(f_conv_h))
    end = time.time()

    print("Time requried to compute = {}s".format(end-start))

    plt.figure(figsize=(10, 10))
    plt.imshow(f_conv_h_1, "gray")
    plt.title("Convolved image 511x511")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(f_conv_h_2, "gray")
    plt.title("Convolved image 512x512")
    plt.show()

    subarray = np.hstack((f_conv_h_2[0:511, 0:255], f_conv_h_2[0:511, 256::]))
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(f_conv_h_1, "gray")
    plt.title("Convolved image 511x511")

    plt.subplot(122)
    plt.imshow(subarray, "gray")
    plt.title("Convolved image 512x512 subarray of 511x511")
    plt.show()

    pdb.set_trace()
