import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pdb
from collections import Counter


def mean_filter(img, k):
    m, n = img.shape
    pad_width = int(k/2)
    img = np.pad(img, pad_width, 'edge')

    new_img = np.zeros((m, n), dtype=np.float)

    for i in range(0 + pad_width, m + pad_width):
        for j in range(0 + pad_width, n + pad_width):
            sum_ = 0
            for p in range(j-pad_width, j+pad_width+1):
                    sum_ += img[i-pad_width:i+pad_width+1:, p].sum()
            new_img[i - pad_width, j - pad_width] = sum_

    scale = (1.0/k**2) * (255 / (np.max(new_img) - np.min(new_img)))
    new_img = scale*(new_img - np.min(new_img))
    new_img = new_img.astype(np.uint8)
    return new_img


def fast_mean_filter(img, k):
    m, n = img.shape
    pad_width = int(k/2)
    img = np.pad(img, pad_width, 'edge')

    new_img = np.zeros(img.shape)

    for i in range(0 + pad_width, m + pad_width):
        sum_value = np.zeros(k)
        for j in range(0 + pad_width, n + pad_width):
            if j == pad_width:
                for p in range(j-pad_width, j+pad_width+1):
                    sum_value[p] = img[i-pad_width:i+pad_width+1:, p].sum()
                new_img[i, j] = sum_value.sum()
                sum_value[0:-1] = sum_value[1:]
                sum_value[-1] = 0
            else:
                sum_value[-1] = img[i-pad_width:i+pad_width+1, j+pad_width].sum()
                new_img[i, j] = sum_value.sum()
                sum_value[0:-1] = sum_value[1:]
                sum_value[-1] = 0

    scale = (1.0/k**2) * (255 / (np.max(new_img) - np.min(new_img)))
    new_img = scale*(new_img - np.min(new_img))
    new_img = new_img.astype(np.uint8)
    return new_img


def median_filter(img, k):
    x, y = img.shape
    pad_width = int(k/2)
    img = np.pad(img, pad_width, 'edge')
    new_img = np.zeros(img.shape)

    for i in range(pad_width, x+pad_width):
        for j in range(pad_width, y+pad_width):
            arr = img[i-pad_width: i+pad_width+1, j-pad_width:j+pad_width+1].flatten()
            index = len(arr)//2
            for p in range(k*k):
                for q in range(0, k*k-i-1):
                    if arr[q] > arr[q+1]:
                        arr[q], arr[q+1] = arr[q+1], arr[q]
            new_img[i, j] = arr[index]

    scale = (1.0/k**2) * (255 / (np.max(new_img) - np.min(new_img)))
    new_img = scale*(new_img - np.min(new_img))
    new_img = new_img.astype(np.uint8)
    return new_img


def fast_median_filter(img, k):
    x, y = img.shape
    pad_width = int(k/2)
    img = np.pad(img, pad_width, 'edge')
    new_img = np.zeros(img.shape)

    curr_median = None
    N = k*k
    med_ind = N//2 + 1
    for i in range(pad_width, x+pad_width):
        local_hist = Counter([])
        for j in range(pad_width, y+pad_width):
            im_slice = img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            if j == pad_width:
                local_hist += Counter(im_slice.flatten())
            else:
                local_hist += Counter(im_slice[:, 2].flatten())

            count = 0
            for each in sorted(local_hist.items()):
                key, val = each
                count += val
                if count >= med_ind:
                    curr_median = key
                    new_img[i, j] = curr_median
                    break
            local_hist -= Counter(im_slice[:, 0].flatten())

    scale = (1.0/k**2) * (255 / (np.max(new_img) - np.min(new_img)))
    new_img = scale*(new_img - np.min(new_img))
    new_img = new_img.astype(np.uint8)
    return new_img
