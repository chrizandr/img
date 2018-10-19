import numpy as np
import pdb
from skimage.color import rgb2lab
from skimage.io import imread
import matplotlib.pyplot as plt


def HistogramEq(img, param="k"):
    range_ = range(0, 256)
    max_val = 255

    if param == "s" or param == "i":
        img = (img * 255).astype(int)
        histogram, bins = np.histogram(img, bins=list(range(0, 257)))
    elif param == "k":
        histogram, bins = np.histogram(img, bins=list(range(0, 257)))
    elif param == "h":
        img = (img * 180 / np.pi).astype(int)
        range_ = range(0, 361)
        max_val = 361
        histogram, bins = np.histogram(img, bins=list(range(0, 362)))

    histogram = histogram/histogram.sum()
    cdf = histogram.cumsum()

    transformation = (max_val * cdf).astype(int)

    m, n = img.shape
    Y = np.zeros_like(img)

    for i in range_:
        pixels = (img == i).nonzero()
        Y[pixels] = transformation[i]

    if param == "s" or param == "i":
        Y = Y/255.0
    if param == "h":
        Y = Y * np.pi / 180

    return Y


def hsi2rgb(img):
    rgb_img = np.zeros(img.shape)

    for l in range(img.shape[0]):
        for m in range(img.shape[1]):
            h, s, i = img[l, m]
            rgb_img[l, m, :] = HSI2RGB(h, s, i)
    return rgb_img * 255


def HSI2RGB(h, s, i):
    h = float(h)
    s = float(s)
    i = float(i)

    x = i * (1 - s)
    if h < 2 * np.pi / 3:
        y = i * (1 + (s * np.cos(h)) / (np.cos(np.pi / 3 - h)))
        z = 3 * i - (x + y)
        r = y
        g = z
        b = x

    elif h < 4 * np.pi / 3:
        y = i * (1 + (s * np.cos(h - 2 * np.pi / 3)) / (np.cos(np.pi / 3 - (h - 2 * np.pi / 3))))
        z = 3 * i - (x + y)
        r = x
        g = y
        b = z
    else:
        y = i * (1 + (s * np.cos(h - 4 * np.pi / 3)) / (np.cos(np.pi / 3 - (h - 4 * np.pi / 3))))
        z = 3 * i - (x + y)
        r = z
        g = x
        b = y

    return r, g, b


def rgb2hsi_color(r, g, b):
    r = float(r)/255
    g = float(g)/255
    b = float(b)/255

    i = (r + g + b) / 3
    if i == 0:
        return 0, 0, 0
    s = 1 - (3 * min([r, g, b])) / (r + g + b)
    h = 0.5 * ((r - g) + (r - b)) / np.sqrt(((r - g)*(r - g)) + ((r - b)*(g - b)))
    h = np.arccos(h)

    if b > g:
        h = 360 * (np.pi / 180) - h

    return h, s, i


def rgb2hsi(img):
    hsi_space = np.zeros(img.shape).astype(float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]
            hsi_space[i, j] = rgb2hsi_color(r, g, b)
    return hsi_space


def rgb2cmy(img):
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    cmy_space = np.zeros(img.shape)
    cmy_space[:, :, 0] = 255 - r_channel
    cmy_space[:, :, 1] = 255 - g_channel
    cmy_space[:, :, 2] = 255 - b_channel

    return cmy_space


def show_plane(plane, space=["R", "G", "B"]):

    ax = plt.subplot(131)
    ax.imshow(plane[:, :, 0], 'gray')
    ax.set_title(space[0] + " plane")

    ax = plt.subplot(132)
    ax.imshow(plane[:, :, 1], 'gray')
    ax.set_title(space[1] + " plane")

    ax = plt.subplot(133)
    ax.imshow(plane[:, :, 2], 'gray')
    ax.set_title(space[2] + " plane")

    # plt.title("".join(space))
    plt.show()


def color_8bit(img):
    new_img = np.zeros(img.shape)
    r_channel = img[:, :, 0].astype(float)
    g_channel = img[:, :, 1].astype(float)
    b_channel = img[:, :, 2].astype(float)

    r_channel = r_channel / 255
    g_channel = g_channel / 255
    b_channel = b_channel / 255

    r_channel = ((r_channel * (2**3 - 1)).astype(int)).astype(float)
    r_channel = ((r_channel / (2**3 - 1)) * 255).astype(int)

    g_channel = ((g_channel * (2**3 - 1)).astype(int)).astype(float)
    g_channel = ((g_channel / (2**3 - 1)) * 255).astype(int)

    b_channel = ((b_channel * (2**2 - 1)).astype(int)).astype(float)
    b_channel = ((b_channel / (2**2 - 1)) * 255).astype(int)

    new_img[:, :, 0] = r_channel
    new_img[:, :, 1] = g_channel
    new_img[:, :, 2] = b_channel

    return new_img


if __name__ == "__main__":
    img = imread("A4_resources/color_bars.tif")
    img = imread("A4_resources/peppers_color.tif")
    img = imread("A4_resources/mandril_color.tif")

    show_plane(img)

    cmy_space = rgb2cmy(img)
    show_plane(cmy_space, ["C", "M", "Y"])

    hsi_space = rgb2hsi(img)
    show_plane(hsi_space, ["H", "S", "I"])

    lab_space = rgb2lab(img)
    show_plane(lab_space, ["L", "a", "b"])

    cimage = color_8bit(img)

    ax = plt.subplot(121)
    ax.imshow(img)
    ax.set_title("Original Image")

    ax = plt.subplot(122)
    ax.imshow(cimage)
    ax.set_title("8 bit safe color image")
    plt.show()

    eqimg = np.zeros(img.shape)
    r_channel = HistogramEq(img[:, :, 0])
    g_channel = HistogramEq(img[:, :, 1])
    b_channel = HistogramEq(img[:, :, 2])
    eqimg[:, :, 0] = r_channel
    eqimg[:, :, 1] = g_channel
    eqimg[:, :, 2] = b_channel

    plt.imshow(eqimg, label="RGB Histogram Equalized")
    plt.show()

    eqimg = np.zeros(img.shape)
    hsi_img = rgb2hsi(img)
    h_channel = HistogramEq(hsi_img[:, :, 0], param="h")
    s_channel = HistogramEq(hsi_img[:, :, 1], param="s")
    i_channel = HistogramEq(hsi_img[:, :, 2], param="i")
    eqimg[:, :, 0] = h_channel
    eqimg[:, :, 1] = s_channel
    eqimg[:, :, 2] = i_channel

    eqimg = hsi2rgb(eqimg)

    plt.imshow(eqimg, label="HSI Histogram Equalized")
    plt.show()

    eqimg = np.zeros(img.shape)
    hsi_img = rgb2hsi(img)
    h_channel = hsi_img[:, :, 0]
    s_channel = hsi_img[:, :, 1]
    i_channel = HistogramEq(hsi_img[:, :, 2], param="i")
    eqimg[:, :, 0] = h_channel
    eqimg[:, :, 1] = s_channel
    eqimg[:, :, 2] = i_channel

    eqimg = hsi2rgb(eqimg)
    plt.imshow(eqimg, label="HSI Histogram Equalized")
    plt.show()
