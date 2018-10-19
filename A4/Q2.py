import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.morphology import convex_hull_image
import pdb


if __name__ == "__main__":
    img = imread("A4_resources/q2.jpg", as_gray=True)
    bimg = (img > threshold_otsu(img)).astype(int)

    mask = label(bimg)

    edge_based = []
    non_overlap = []
    overlap = []

    for l in np.unique(mask)[1::]:
        x, y = (mask == l).nonzero()

        if 0 in x or 0 in y or bimg.shape[0]-1 in x or bimg.shape[1]-1 in y:
            edge_based.append((x, y))

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        region = bimg[x_min-1:x_max+2, y_min-1:y_max+2]
        convex_hull = convex_hull_image(region)

        if np.all(region == convex_hull):
            non_overlap.append((x, y))
        else:
            overlap.append((x, y))

    edge = np.zeros_like(bimg)
    over = np.zeros_like(bimg)
    nover = np.zeros_like(bimg)
    for x, y in edge_based:
        edge[x, y] = 1
    for x, y in overlap:
        over[x, y] = 1
    for x, y in non_overlap:
        nover[x, y] = 1

    pdb.set_trace()
