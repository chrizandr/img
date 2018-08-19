"""Functions and definitions for Q1."""
import matplotlib.pyplot as plt
import numpy as np
import pdb
from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb


def resize_image(img1, img2):
    """Resize two images to the same sizes."""
    m, n, _ = img1.shape
    i, j, _ = img2.shape
    h = max(m, i)
    w = max(n, j)
    img1 = resize(img1, (h, w), anti_aliasing=True)
    img2 = resize(img2, (h, w), anti_aliasing=True)
    return img1, img2


def showImages(fg, bg, result):
    """Plot images."""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(131)
    ax.imshow(fg)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Foreground")

    ax = fig.add_subplot(132)
    ax.imshow(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Background")

    ax = fig.add_subplot(133)
    ax.imshow(result)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("Result")

    return fig


def ChromaKey(fg, bg, keyColor, threshold=200):
    """Replace keyColor in fg with bg."""
    color_diff = fg.astype(np.float) - keyColor.astype(np.float)
    color_dist = np.sqrt(np.sum(color_diff * color_diff, axis=2))
    color_pixels = (color_dist > threshold).nonzero()

    transform_map = np.zeros(fg.shape, dtype=np.uint8)
    transform_map[color_pixels] = 1
    inverse_map = 1 - transform_map

    transform_map = transform_map * fg
    inverse_map = inverse_map * bg
    final_image = transform_map + inverse_map

    return final_image


if __name__ == "__main__":
    fg = imread("A1_resources/ck_1.jpg", as_gray=False)
    bg = imread("A1_resources/bg.png", as_gray=False)
    fg, bg = resize_image(fg, bg)
    # fg = img_as_ubyte(rgba2rgb(fg))
    bg = img_as_ubyte(rgba2rgb(bg))
    fg = img_as_ubyte(fg)
    bg = img_as_ubyte(bg)

    keyColor = np.array([0, 255, 0], dtype=np.uint8)

    result = ChromaKey(fg, bg, keyColor, 170)
    fig = showImages(fg, bg, result)
    fig.show()
    pdb.set_trace()
