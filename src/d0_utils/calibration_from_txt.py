"""
This module serves to calibrate images of a video thanks to a txt file.
"""
import numpy as np

# Exceptions
from src.d0_utils.extractions.exceptions.exception_classes import FindPathExtractError

# To extract images from a video
from src.d0_utils.extractions.extract_image import extract_image_video

# To transform image to get the top-down view
from src.d0_utils.perspective_correction.perspective_correction import get_top_down_image


def calibrate_from_txt(path_video, path_txt, time_begin=0, time_end=-1):
    """
    Extract the images and transform them to get the top-down view.

    Args:
        path_video (WindowsPath): the path that leads to the video.

        path_txt (WindowsPath): the path that leads to the txt file for the calibration.

        time_begin (integer): the beginning time in second.
            Default value = 0

        time_end (integer): the ending time in second.
            Default value = -1
            if time_end == -1, the video is viewed until the end.

    Returns:
        list_images (list of array): list of the calibrated images.
    """
    # Check that the paths exists
    if not path_video.exists():
        raise FindPathExtractError(path_video)
    if not path_txt.exists():
        raise FindPathExtractError(path_txt)

    # Get the homography
    file = open(path_txt, 'r')
    lines = file.readlines()
    homography = np.fromstring(lines[-2], dtype=float, sep=',')
    file.close()

    homography = np.reshape(homography, (3, 3))

    # Get the images
    list_images = extract_image_video(path_video, time_begin, time_end)
    nb_images = len(list_images)

    # Transform the images
    for index_image in range(nb_images):
        list_images[index_image] = get_top_down_image(list_images[index_image], homography)

    return list_images
