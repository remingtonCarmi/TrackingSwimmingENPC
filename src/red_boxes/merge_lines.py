"""
Allows to merge the lines-separated frames into a unique frame
"""

from crop_lines import *
import matplotlib.pyplot as plt
import numpy as np


def bgr_to_rgb(image):
    """
    To convert a BGR image into a RGB one
    Args:
        image(numpy array): image to convert

    Returns:
        im(numpy array): new image, in RGB
    """
    im = image.copy()
    im[:, :, 0], im[:, :, 2] = image[:, :, 2], image[:, :, 0]
    return im


def merge(images):
    """
    To vertically merge 10 images into a single image. The first image will be on the top.
    Args:
        images (list of numpy arrays): list of images, all with the same length, not necessarily the same high

    Returns:
        merged_image(numpy array): the merged image
    """
    merged_image = images[0]
    for line in images[1:]:
        merged_image = np.concatenate((merged_image, line), axis=0)
    return merged_image


if __name__ == "__main__":
    LINES, POINTS = load_lines("frame2.jpg", "vid0_clean", 0, 0)

    MERGED = merge(LINES)

    plt.imshow(bgr_to_rgb(MERGED))
    plt.show()
