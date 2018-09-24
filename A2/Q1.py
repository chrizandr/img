import numpy as np
import pdb
from skimage import io
import matplotlib.pyplot as plt


def filterImg(img, kernel, padding=True):
    m, n = img.shape
    pad_width = int(kernel.shape[0]/2) if padding else 0
    img = np.pad(img, pad_width, 'edge')

    new_img = np.zeros((m, n))

    for i in range(0 + pad_width, m + pad_width):
        for j in range(0 + pad_width, n + pad_width):
            region = img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            value = region * kernel
            new_img[i - pad_width, j - pad_width] = value.sum()

    clip_top = (new_img > 1).nonzero()
    new_img[clip_top] = 1
# scale = 255/(np.max(new_img) - np.min(new_img))
# new_img = scale*(new_img - np.min(new_img))
    return new_img


if __name__ == "__main__":
    im = io.imread('A2_resources/sky2.jpg', as_gray=True)
    kernel = 1.0/16*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    img = filterImg(im, kernel)
