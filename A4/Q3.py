import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.neighbors import KNeighborsClassifier as KNN
from skimage.morphology import binary_opening
import matplotlib.pyplot as plt
import pdb


def get_histogram_color(img):
    m, n, k = img.shape
    colors = dict()
    for i in range(m):
        for j in range(n):
            color = tuple(img[i, j])
            colors[color] = colors.get(color, 0) + 1
    return list(colors.values()), list(colors.keys())


def segment_color(img, colors, bins):
    m, n, k = img.shape
    mask = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            c = tuple(img[i, j])
            index = np.where((bins == c).all(axis=1))[0][0]
            mask[i, j] = colors[index]
    return mask


def get_components(mask):
    colors = np.unique(mask)
    components = []
    for color in colors:
        bin_mask = np.zeros(mask.shape)
        indices = (mask == color).nonzero()
        bin_mask[indices] = 1
        components.append(bin_mask)
    return components


def remove_nonzero(nbs):
    s = []
    for i in nbs:
        if i != 0:
            s.append(i)
    return s


def twopass_segmentation(img):
    relations = {}
    m, n = img.shape
    color = 1
    mask = np.zeros((m, n))
    indices = img.nonzero()

    # Pass 1
    for k in range(len(indices[0])):
        i, j = indices[0][k], indices[1][k]

        if i == 0 and j == 0:
            nbs = []
        elif i == 0:
            nbs = remove_nonzero([mask[i, j-1], mask[i+1, j-1]])
        elif j == 0:
            nbs = remove_nonzero([mask[i-1, j]])
        elif i == m-1:
            nbs = remove_nonzero([mask[i-1, j], mask[i-1, j-1], mask[i, j-1]])
        else:
            nbs = remove_nonzero([mask[i-1, j], mask[i-1, j-1], mask[i, j-1], mask[i+1, j-1]])

        if len(nbs) > 0:
            c = min(nbs)
            mask[i, j] = c
            for n in nbs:
                if n != c:
                    relations[(c, n)] = 1
        else:
            mask[i, j] = color
            color += 1

    # Pass 2
    for r in relations.keys():
        min_c, max_c = r
        indices = (mask == max_c).nonzero()
        mask[indices] = min_c

    return mask


def twopass_segmentation_4connect(img):
    relations = {}
    m, n = img.shape
    color = 1
    mask = np.zeros((m, n))
    indices = img.nonzero()

    # Pass 1
    for k in range(len(indices[0])):
        i, j = indices[0][k], indices[1][k]

        if i == 0 and j == 0:
            nbs = []
        elif i == 0:
            nbs = remove_nonzero([mask[i, j-1]])
        elif j == 0:
            nbs = remove_nonzero([mask[i-1, j]])
        elif i == m-1:
            nbs = remove_nonzero([mask[i-1, j], mask[i, j-1]])
        else:
            nbs = remove_nonzero([mask[i-1, j], mask[i, j-1]])

        if len(nbs) > 0:
            c = min(nbs)
            mask[i, j] = c
            for n in nbs:
                if n != c:
                    relations[(c, n)] = 1
        else:
            mask[i, j] = color
            color += 1

    # Pass 2
    for r in relations.keys():
        min_c, max_c = r
        indices = (mask == max_c).nonzero()
        mask[indices] = min_c

    return mask


if __name__ == "__main__":
    img = imread("A4_resources/q3.png")

    hist, bins = get_histogram_color(img)
    bins = np.array(bins)
    hist = np.array(hist)

    sorted_vals = np.flip(hist.argsort(), axis=0)

    centers = bins[sorted_vals[0:7]]
    labels = np.arange(7)

    model = KNN(n_neighbors=1)
    model.fit(centers, labels)

    colors = model.predict(bins)
    mask = segment_color(img, colors, bins)

    components = get_components(mask)

    for c in components[1::]:
        cimage = binary_opening(c).astype(int)
        mask = twopass_segmentation(cimage)
        print(len(np.unique(mask)))

    img = imread("A4_resources/q3_2.jpg", as_gray=True)
    img = (img > threshold_otsu(img)).astype(int)
    mask = twopass_segmentation_4connect(img)

    m, n = img.shape
    l, k = 8, 10

    valid = []
    for i in range(0, m-l+1, l):
        for j in range(0, n-k+1, k):
            region = mask[i:i+l, j:j+k]
            if len(np.unique(region)) == 3:
                plt.imshow(region)
                plt.show()
                valid.append(region)

    pdb.set_trace()
