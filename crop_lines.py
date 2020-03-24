"""
This code separates an image in several images, delimited by the water lines
"""

import os
import shutil

# from correction_perspective import correctPerspectiveImg
from extract_image import extract_image_video


def slope_intercept(a, b):
    """
    USELESS AT THE MOMENT
    Args:
        a:
        b:

    Returns:

    """
    ya, xa = a
    yb, xb = b

    k = (yb - ya) / (xb - xa)
    return k, ya - k*xa


def crop(image, list_y):
    """
    To vertically crop an images at given positions
    Args:
        image(numpy array): image to crop
        list_y(list of integers): list of vertical positions where we crop

    Returns:
        images_crop(list of numpy arrays): list of cropped images

    """

    images_crop = []

    y_prev = list_y[0]

    for y in list_y[1:]:
        images_crop.append(image[y_prev: y, :])
        y_prev = y

    return images_crop


def load_lines(frame_name, folder, video_name, time_begin, time_end):
    frame_name = folder + frame_name

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    list_images = extract_image_video("videos\\" + video_name, time_begin, time_end, True, folder)
    # plt.imsave(frame_name, i[0])

    list_images_crop = []
    # image = cv2.imread(frame_name)

    points = [6, 82, 156, 233, 309, 382, 458, 531, 604, 679, 749]
    # points = [3, 76, 153, 234, 306, 380, 456, 531, 604, 680, 749]

    for im in list_images:
        list_images_crop.append(crop(im, points))

    # images_crop = crop(image, points)

    return list_images_crop, points


if __name__ == "__main__":
    FOLDER = "test\\red_boxes\\"
    LIST, POINTS = load_lines("frame2.jpg", FOLDER, "vid0_clean", 0, 0)
