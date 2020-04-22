"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import cv2
import numpy as np
import random as rd
from src.utils.extract_image import extract_image_video
from src.utils.extract_image import TimeError
from src.utils.exception_classes import VideoFindError
from src.utils.make_video import make_video
from src.utils.point_selection.point_selection import perspective_selection
from src.utils.perspective_correction.perspective_correction import correct_perspective_img, get_perspective_matrix
import matplotlib.pyplot as plt


def transform_in_2d(points_pixels, points_pool, image):
    (height, width) = image.shape[: 2]

    # Get the corrected image in the entire pool
    points_pool_pixel = np.zeros((4, 2))
    points_pool_pixel[:, 0] = (points_pool[:, 0] + 1) * width / 52
    points_pool_pixel[:, 1] = points_pool[:, 1] * height / 25

    # Get the transformation
    perspective_matrix = get_perspective_matrix(points_pixels, points_pool_pixel)
    corrected_image = correct_perspective_img(image, perspective_matrix)

    # Find the first column that is not black
    index_w = 0
    while index_w < width and np.sum(corrected_image[:, index_w]) == 0:
        index_w += 1
    left_column = index_w

    # Find the last column that is not black
    index_w = width - 1
    while index_w >= 0 and np.sum(corrected_image[:, index_w]) == 0:
        index_w -= 1
    right_column = index_w

    # PUT AN EXCEPTION IF LEFT_COLUMN > RIGHT_COLUMN
    # print("Left_point", left_column * 52 / width)
    # print("Right_point", right_column * 52 / width)

    # Get the final transformation
    points_pool_pixel[:, 0] = (points_pool_pixel[:, 0] - left_column) / (right_column - left_column) * width

    return points_pool_pixel


def calibrate_video(name_video, time_begin=0, time_end=-1, destination_video=Path("../output/test/")):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        name_video (string): name of the video.

        time_begin (integer): the starting time in second. Default value = 0.

        time_end (integer): the ending time in second. Default value = -1.

        destination_video (string): the destination path of the cleaned video
            Default value = "".
    """
    # Get the images
    print("Get the images ...")
    list_images = extract_image_video(name_video, time_begin, time_end)
    nb_images = len(list_images)

    # Selection of the perspective points on a random image
    print("Point selection ...")
    (points_image, points_real) = perspective_selection(list_images[rd.randint(int(nb_images / 10), int(nb_images / 5))])

    # Get the real points in pixels
    points_real_pixel = transform_in_2d(points_image, points_real, list_images[0])

    # Get the perspective matrix
    perspective_matrix = get_perspective_matrix(points_image, points_real_pixel)

    # Transform the images
    print("Correction of images ...")
    for index_image in range(nb_images):
        list_images[index_image] = correct_perspective_img(list_images[index_image], perspective_matrix)

    # Get the fps
    video = cv2.VideoCapture(str(name_video))
    fps_video = int(video.get(cv2.CAP_PROP_FPS))

    print("Make the corrected video ...")
    corrected_video = "corrected_" + name_video.parts[-1]
    make_video(corrected_video, list_images, fps_video, destination_video)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid1.mp4")
    DESTINATION_VIDEO = Path("../data/videos/corrected/")
    try:
        calibrate_video(PATH_VIDEO, 10, 11, DESTINATION_VIDEO)
    except TimeError as time_error:
        print(time_error.__repr__())
    except VideoFindError as video_find_error:
        print(video_find_error.__repr__())
