"""
This code separates an image in several images, delimited by the water lines
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from red_spectrum import load_image
from detection import select_points, register_points
# from correction_perspective import correctPerspectiveImg
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
    frame_name = "test\\t\\" + frame_name
    if not os.path.exists(frame_name):
        i = extract_image_video("videos\\" + video_name, time_begin, time_end, True, "test\\t\\")
        #plt.imsave(frame_name, i[0])

    list_images_crop = []
    #image = cv2.imread(frame_name)

    points = [6, 82, 156, 233, 309, 382, 458, 531, 604, 679, 749]
    #points = [3, 76, 153, 234, 306, 380, 456, 531, 604, 680, 749]
    for im in i:
        list_images_crop.append(crop(im, points))

    #images_crop = crop(image, points)

    return list_images_crop, points


if __name__ == "__main__":
    load_lines("frame2.jpg", "vid0_clean", 0, 0)
