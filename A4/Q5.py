import numpy as np
import pdb
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    output_images = []
    for i in range(60, 160, 20):
        img = np.zeros((100, 100, 3))
        h = (np.pi * i) / 180

        r, g, b = HSI2RGB(h, 1, 1)
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        output_images.append(img)

    fig, ax = plt.subplots(4, 5)
    for i in range(len(output_images)):
        ax[i//5][i % 5].imshow(output_images[i])

    plt.show()
    pdb.set_trace()
