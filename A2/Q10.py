import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pdb


def bilateral_filter(region, sigma_d, sigma_r):
    size = region.shape[0]
    i = int(size/2)
    j = i

    f_ij = region[i, j]
    bi_filter = np.zeros((size, size))

    for k in range(size):
        for l in range(size):
            domain_term = ((i-k)**2 + (j-l)**2)/(2*sigma_d**2)
            range_term = ((f_ij-region[k, l])**2)/(2*sigma_r**2)
            bi_filter[k, l] = np.exp(-domain_term-range_term)

    return bi_filter


def bilateral_filtering(img, size=9, sigma_d=8, sigma_r=0.05):
    x, y = img.shape
    pad_width = int(size/2)
    img = np.pad(img, pad_width, 'edge')
    new_img = np.zeros(img.shape)

    for i in range(pad_width, x+pad_width):
        for j in range(pad_width, y+pad_width):
            region = img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            kernel = bilateral_filter(region, sigma_d, sigma_r)
            kernel = kernel/kernel.sum()
            new_img[i, j] = (region*kernel).sum()

    new_img = new_img[pad_width: x+pad_width, pad_width:y+pad_width]
    return new_img


def high_boost_filtering(img, size=5, A=2):
    x, y = img.shape
    pad_width = int(size/2)
    img = np.pad(img, pad_width, 'edge')
    new_img = np.zeros(img.shape)

    hfilter = -np.ones((size, size))
    hfilter[pad_width, pad_width] = (size**2)*A - 1
    for i in range(pad_width, x+pad_width):
        for j in range(pad_width, y+pad_width):
            im_slice = img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            new_img[i, j] = (im_slice*hfilter).sum()

    new_img = new_img[pad_width: x+pad_width, pad_width:y+pad_width]
    return new_img
