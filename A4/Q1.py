import numpy as np
import pdb
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def morph_op(img, selems):
    m, n = img.shape
    out = img.copy()
    for selem in selems:
        l, k = selem.shape

        for i in range(l-1, m-l+1):
            for j in range(k-1, n-k+1):
                region = img[i:i+l, j:j+k]
                if np.all(region == selem):
                    out[i + l//2, j + k//2] = 0

    return out


def get_selems(selem):
    dont_cares = (selem == -1).nonzero()
    num_dc = len(dont_cares[0])
    num_selems = 2**num_dc
    selems = []

    for i in range(num_selems):
        bin_rep = [int(x) for x in bin(i)[2:]]
        if len(bin_rep) != num_dc:
            bin_rep = [0 for x in range(num_dc - len(bin_rep))] + bin_rep

        print(bin_rep)
        s_new = selem.copy()
        s_new[dont_cares] = bin_rep
        selems.append(s_new)

    return selems


def rotate_selem(selem, angle):
    if angle == 0:
        return selem
    elif angle == 90:
        return np.rot90(selem)
    elif angle == 180:
        return np.rot90(selem, k=2)
    elif angle == 270:
        return np.rot90(selem, k=3)


if __name__ == "__main__":
        img = imread("A4_resources/q1_4.jpg", as_gray=True)
        I1 = (img > threshold_otsu(img)).astype(int)
        I9 = np.zeros(I1.shape)

        selem_1 = np.array([[0, 0, 0],
                            [-1, 1, -1],
                            [1, 1, 1]])
        S1_0 = get_selems(selem_1)

        selem_2 = np.array([[-1, 0, 0],
                            [1, 1, 0],
                            [-1, 1, -1]])
        S2_0 = get_selems(selem_2)

        S1_90 = [rotate_selem(s, 90) for s in S1_0]
        S2_90 = [rotate_selem(s, 90) for s in S2_0]

        S1_180 = [rotate_selem(s, 180) for s in S1_0]
        S2_180 = [rotate_selem(s, 180) for s in S2_0]

        S1_270 = [rotate_selem(s, 270) for s in S1_0]
        S2_270 = [rotate_selem(s, 270) for s in S2_0]

        i = 0
        while not np.all(I1 == I9):
            if i != 0:
                I1 = I9
            print("Iteration : ", i)
            I2 = morph_op(I1, S1_0)
            I3 = morph_op(I2, S2_0)
            I4 = morph_op(I3, S1_90)
            I5 = morph_op(I4, S2_90)
            I6 = morph_op(I5, S1_180)
            I7 = morph_op(I6, S2_180)
            I8 = morph_op(I7, S1_270)
            I9 = morph_op(I8, S2_270)
            imsave("output_q1_iter" + str(i) + ".png", I9*255)
            i += 1

        pdb.set_trace()
