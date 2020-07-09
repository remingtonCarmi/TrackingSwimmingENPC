"""
This module creates a folder filled with LANES taken from a video.
"""
import os
import cv2
from pathlib import Path

# Exceptions
from src.d0_utils.extractions.exceptions.exception_classes import TimeError
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError

# To calibrate an image from a txt file
from src.d0_utils.calibration_from_txt import calibrate_from_txt

# To split the image into LANES and save them
from src.d0_utils.split_and_save_data.split_image import split_and_save


def create_data(path_video, path_txt, margin, time_begin=0, time_end=-1, nb_lines=10, destination=None):
    """
    Fill the destination path with pictures of the LANES taken from the video.

    Args:
        path_video (WindowsPath): path to the video

        path_txt (WindowsPath): path to the txt file to calibrate the video

        margin (integer): number of lines of pixels to add to LANES
            - we remove (if margin > 0)
            - we add (if margin < 0)

        time_begin (integer): the beginning time
            Default value = -1

        time_end (integer): the ending time
            Default value = -1
            if time_end == -1, the video is viewed until the end.

        nb_lines (integer): the number of LANES in the pool
            Default value = 10

        destination (integer): the path where the LANES will be saves
            Default value = None
    """
    if destination is None:
        destination = Path("../data/1_intermediate_top_down_lanes/LANES/tries")

    # Verify that the paths exists
    if not path_video.exists():
        raise FindPathError(path_video)
    if not path_txt.exists():
        raise FindPathError(path_txt)
    if not destination.exists():
        raise FindPathError(destination)

    # Get the name of the directory
    name_video = path_video.parts[-1][: -4]
    path_directory = Path(str(destination / name_video))

    # Check if the data has already been generated
    if path_directory.exists():
        raise AlreadyExistError(path_directory)
    else:
        # Create the directory where to save the data
        os.makedirs(path_directory)

    # Save the image second per second
    for time in range(time_begin, time_end):
        # Calibrate and extract image from the corrected video
        corrected_images = calibrate_from_txt(path_video, path_txt, time, time + 1)

        # fps = frame per second
        fps = len(corrected_images)

        # Save the data
        for idx_image in range(fps):
            # Split and save the image
            split_and_save(corrected_images[idx_image], margin, path_directory, time * fps + idx_image, nb_lines)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/0_raw_videos/DSC_6980.mp4")
    PATH_TXT = Path("../data/1_intermediate_top_down_lanes/calibration/tries/DSC_6980.txt")

    MARGIN = 0

    try:
        create_data(PATH_VIDEO, PATH_TXT, MARGIN, time_begin=0, time_end=1)
    except FindPathError as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as already_exists:
        print(already_exists.__repr__())
    except TimeError as time_error:
        print(time_error.__repr__())
