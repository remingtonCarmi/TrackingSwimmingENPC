""" main for red boxes"""

import os
import shutil
import cv2
import numpy as np
from time import time
from PIL import ImageDraw
import matplotlib.pyplot as plt

from src.extract_image import extract_image_video
from src.perspective.correction_perspective import correct_perspective_img

from src.red_boxes.crop_lines import crop_list
from src.red_boxes.all_rectangles import boxes_list_images
from src.red_boxes.all_rectangles import plot_length_rectangles


def clean_test(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


src = np.float32([(48., 290.),
                 (1071., 150.),
                 (569., 794.),
                 (1844., 536.)]
                 )


dst2 = np.float32([(1500, 0),
                   (0, 0),
                   (1500, 750),
                   (0, 750)]
                  )


def draw_rectangle(image, x0, y0, size_x, size_y, outline=5):
    im = np.copy(image)

    for k in range(size_x):
        for y in range(0, outline):
            im[y0 - y, x0 + k] = [255, 0, 0]
            im[y0 + y + size_y, x0 + k] = [255, 0, 0]
    for k in range(size_y):
        for x in range(-(outline//2), outline//2 + 1):
            im[y0 + k, x0 - x] = [255, 0, 0]
            im[y0 + k, x0 + x + size_x] = [255, 0, 0]

    return im


if __name__ == "__main__":
    # FOLDER = "..\\..\\test\\red_boxes\\"
    # FRAME_NAME = FOLDER + "frame2.jpg"
    # clean_test(FOLDER)
    t = time()
    LIST_IMAGES = extract_image_video("..\\data\\videos\\vid0", 12, 20, False)
    NB_IMAGES = len(LIST_IMAGES)
    # IM0 = extract_image_video("..\\data\\videos\\vid0", 0, 0, False)[0]
    # IM0_CORR = cv2.cvtColor(correct_perspective_img(IM0, src, dst2, True, False), cv2.COLOR_BGR2RGB)
    # plt.imshow(IM0_CORR)
    # plt.show()

    IMAGES_CORR = []
    for i in range(NB_IMAGES):
        IMAGES_CORR.append(correct_perspective_img(LIST_IMAGES[i], src, dst2, True, False))
        IMAGES_CORR[i] = cv2.cvtColor(IMAGES_CORR[i], cv2.COLOR_BGR2RGB)

    # 748
    # POINTS = [71, 146, 221, 295, 370, 445, 520, 596, 673]
    POINTS = [76, 150, 228, 306, 380, 456, 531, 604, 680]

    MARGIN = 8
    LIST_IMAGES_CROP = crop_list(IMAGES_CORR, POINTS, MARGIN)
    LIST_RECTANGLES = boxes_list_images(LIST_IMAGES_CROP, POINTS)
    # plt.imshow(LIST_IMAGES_CROP[0][6])

    im = np.copy(IMAGES_CORR)
    # draw = draw_rectangle(draw, LIST_RECTANGLES[0][1][0][0],
    #                       LIST_RECTANGLES[0][1][0][1],
    #                       LIST_RECTANGLES[0][1][1][0],
    #                       LIST_RECTANGLES[0][1][1][1]
    #                       )
    for i in range(len(im)):
        for swimmer in LIST_RECTANGLES[i]:
            if swimmer[1][0] < 500:
                im[i] = draw_rectangle(im[i],
                                       swimmer[0][0],
                                       swimmer[0][1] + MARGIN,
                                       swimmer[1][0],
                                       swimmer[1][1],
                                       3)
        # plt.figure()
        # plt.imshow(im[i])

    print(time() - t)
    # 31.282073974609375
    plot_length_rectangles(LIST_RECTANGLES, 2)


