"""
This code separates an image in several images, delimited by the water lines
"""

import os
import shutil

# from correction_perspective import correctPerspectiveImg
from src.utils.extract_image import extract_image_video


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


def crop(image, list_y, margin=0):
    """
    To vertically crop an images at given positions
    Args:
        margin (integer):
        image(numpy array): image to crop
        list_y(list of integers): list of vertical positions where we crop

    Returns:
        images_crop(list of numpy arrays): list of cropped images

    """

    images_crop = []

    y_prev = list_y[0]

    for y in list_y[1:]:
        images_crop.append(image[y_prev + margin: y - margin, :])
        y_prev = y
    return images_crop


def load_lines(frame_name, folder, video_name, time_begin, time_end):
    """

    Args:
        frame_name (string): generic name for the images that will be saved
        folder (string): name of the folder where we save all the images
        video_name: name of the video from where we extract the image (perspective has to be corrected)
        time_begin (float): starting time of the video we consider
        time_end (float): ending time of the video we consider

    Returns:
        list_images_crop (list of list of numpy arrays):
            list_images_crop[i][j] : the j-th water line of the i-th frame from the video
        points (list of integers): manually-selected ordinates of the 11 lines

    """
    frame_name = folder + frame_name

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    list_images = extract_image_video("..\\..\\data\\videos\\" + video_name, time_begin, time_end, True, folder)

    list_images_crop = []

    # manually-selected ordinates of the 11 lines
    points = [6, 82, 156, 233, 309, 382, 458, 531, 604, 679, 749]
    # points = [3, 76, 153, 234, 306, 380, 456, 531, 604, 680, 749]

    for im in list_images:
        list_images_crop.append(crop(im, points))

    return list_images_crop, points


# cleaned function, without imports and call of functions
def crop_list(list_images, points, margin=0):

    list_images_crop = []

    for image in list_images:
        list_images_crop.append(crop(image, points, margin))

    return list_images_crop


if __name__ == "__main__":
    FOLDER = "test\\red_boxes\\"
    LIST, POINTS = load_lines("frame2.jpg", FOLDER, "vid0_clean", 0, 0)
