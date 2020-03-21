"""
This code separates an image in several images, delimited by the water lines
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from red_spectrum import load_image
from detection import select_points, register_points
from correction_perspective import correctPerspectiveImg
from extract_image import extract_image_video


def slope_intercept(a, b):
    """
    Finally useless (probably)
    @param a:
    @param b:
    @return:
    """
    ya, xa = a
    yb, xb = b

    k = (yb - ya) / (xb - xa)
    return k, ya - k*xa


def crop(image, lines):
    """
    lines : [ y1, y2, y3, y4, ..]
    @param image: numpy array
    @param lines:
    @return:
    """

    images_crop = []

    y_prev = lines[0]

    for y in lines[1:]:
        images_crop.append(image[y_prev: y, :])
        y_prev = y

    return images_crop


def load_lines(frame_name, video_name, time_begin, time_end):
    frame_name = "test\\" + frame_name
    if not os.path.exists(frame_name):
        i = extract_image_video("videos\\" + video_name, time_begin, time_end, False)
        plt.imshow(i[0])
        plt.show()
        plt.imsave(frame_name, i[0])
    image = load_image(frame_name)

    points = [6, 82, 156, 233, 309, 382, 458, 531, 604, 679, 749]
    images_crop = crop(image, points)

    return images_crop, points


if __name__ == "__main__":
    load_lines("test1.jpg", "vid0_clean", 0, 1)
