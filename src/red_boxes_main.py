""" main for red boxes"""

import os
import shutil
import cv2
import numpy as np
from time import time
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


src1 = np.float32([(48., 290.),
                 (1071., 150.),
                 (569., 794.),
                 (1844., 536.)]
                 )

src = np.float32([(42., 287.),
                 (1063., 150.),
                 (566., 790.),
                 (1838., 534.)]
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
    LIST_IMAGES = extract_image_video("..\\data\\videos\\vid0", 14, 18, False)
    NB_IMAGES = len(LIST_IMAGES)

    # IM0 = extract_image_video("..\\data\\videos\\vid0", 3, 3, False)[0]
    # IM0_CORR = cv2.cvtColor(correct_perspective_img(IM0, src, dst2, True, False), cv2.COLOR_BGR2RGB)

    IMAGES_CORR = []
    for i in range(NB_IMAGES):
        IMAGES_CORR.append(correct_perspective_img(LIST_IMAGES[i], src, dst2, True, False))
        IMAGES_CORR[i] = cv2.cvtColor(IMAGES_CORR[i], cv2.COLOR_BGR2RGB)

    # POINTS = [71, 146, 221, 295, 370, 445, 520, 596, 673]
    # POINTS = [76, 150, 228, 306, 380, 456, 531, 604, 680]
    POINTS = [75, 156, 231, 306, 380, 455, 530, 603, 677]

    MARGIN = 14
    LIST_IMAGES_CROP = crop_list(IMAGES_CORR, POINTS, MARGIN)
    LIST_RECTANGLES = boxes_list_images(LIST_IMAGES_CROP, POINTS)

    im = np.copy(IMAGES_CORR)

    for i in range(len(im)):
        for swimmer in LIST_RECTANGLES[i]:
            # if swimmer[1][0] < 500:
            im[i] = draw_rectangle(im[i],
                                   swimmer[0][0],
                                   swimmer[0][1] + MARGIN,
                                   swimmer[1][0],
                                   swimmer[1][1],
                                   3)
        # plt.figure()
        # plt.imshow(im[i])

    print("Runtime : ", round(time() - t, 3), " seconds.")
    line = 1
    all_lines = [0, 1, 2, 3, 4, 5, 6, 7]

    check_swimmer_arms = False

    if NB_IMAGES > 10:
        LIST = plot_length_rectangles(LIST_RECTANGLES, all_lines, "area")

        if check_swimmer_arms:
            for i in LIST:
                plt.figure()
                plt.title("Image number " + str(i))
                box_i = LIST_RECTANGLES[i][line]
                plt.imshow(IMAGES_CORR[i][box_i[0][1]: box_i[0][1] + box_i[1][1], box_i[0][0]: box_i[0][0] + box_i[1][0]])
    plt.show()
