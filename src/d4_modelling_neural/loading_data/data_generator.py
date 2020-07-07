"""
This module generate an object that holds the paths to the images
and the labels of these images.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.transformation_tools.exceptions import FindPathDataError


def generate_data(paths_label, starting_data_paths=None, starting_calibration_paths=None, take_all=False):
    """
    Produce a list of list of 3 elements : path_to_image, x_head, y_head.
    We look in the labels and if the image is in the computer, we add it to the list.

    Args:
        paths_label (List of WindowsPath): the list of paths to the labels.

        starting_data_paths (WindowsPath): the beginning of the path to the images.
            Default value = None

        starting_calibration_paths (WindowsPath): the beginning of the path to the calibration file.
            Default value = None

        take_all (boolean): if True, all the images are taken. Otherwise, only the images
            with a positive label is taken.
            Default value = False

    Returns:
        full_data (list of list of 3 elements : (WindowsPath, integer, integer, float):
            list of elements like (path_to_image, x_head, y_head, length_image) for the training
    """
    if starting_data_paths is None:
        starting_data_paths = Path("../../../data/1_intermediate_top_down_lanes/lanes/tries")

    if starting_calibration_paths is None:
        starting_calibration_paths = Path("../../../data/1_intermediate_top_down_lanes/calibration/tries")

    nb_videos = len(paths_label)

    # Construction of data_paths
    paths_data = np.zeros(nb_videos, dtype=Path)
    paths_calibration = np.zeros(nb_videos, dtype=Path)
    for idx_video in range(nb_videos):
        video_name = paths_label[idx_video].parts[-1][: -4]
        # Path for images
        paths_data[idx_video] = starting_data_paths / video_name

        # Path for calibration
        video_name_txt = video_name + ".txt"
        paths_calibration[idx_video] = starting_calibration_paths / video_name_txt

    # Verify that all the paths exists:
    for idx_video in range(nb_videos):
        if not paths_label[idx_video].exists():
            raise FindPathDataError(paths_label[idx_video])
        if not paths_data[idx_video].exists():
            raise FindPathDataError(paths_data[idx_video])
        if not paths_calibration[idx_video].exists():
            raise FindPathDataError(paths_calibration[idx_video])

    full_data = []

    for idx_video in range(nb_videos):
        # Get the labels
        labels = pd.read_csv(paths_label[idx_video])

        # Get the length of the image
        length_image = get_length_image(paths_calibration[idx_video])

        # Get the data
        data = get_full_data(paths_data[idx_video], labels, length_image, take_all)

        # Add the data
        full_data.extend(data)

    return full_data


def get_full_data(path_data, labels, length_image, take_all):
    """
    Produce a list of list of 3 elements : path_to_image, x_head, y_head, length_image.
    We look in the labels and if the image is in the computer, we add it to the list.

    Args:
        path_data (WindowsPath): path to the images.

        labels (DataFrame): table that contains the labels.

        length_image (float): the length of the image in meters.

        take_all (boolean): if True, all the images are taken. Otherwise, only the images
            with a positive label is taken.

    Returns:
        full_data (list of list of 4 elements : (WindowsPath, integer, integer, float):
            list of elements like (path_to_image, x_head, y_head, length_image) for the training.
    """
    full_data = []
    for ((lane, frame), label) in labels.iterrows():
        # Add to the list only if the image has been labeled with a right position
        if label[0] >= 0 or take_all:
            name_image = "l{}_f{}.jpg".format(lane, str(frame).zfill(4))
            path_image = path_data / name_image
            # Add to the list only if the image is in the computer
            if path_image.exists():
                full_data.append([path_image, label[0], label[1], length_image])

    return full_data


def get_length_image(path_calibration):
    """
    Get the length of the video in meters.

    Args:
        path_calibration (WindowsPath): the path that leads to the calibration file of the video.

    Returns:
        (float): the length of the video in meters.
    """
    file = open(path_calibration, 'r')
    lines = file.readlines()
    extreme_values = np.fromstring(lines[-1], dtype=float, sep=',')

    return abs(extreme_values[2] - extreme_values[0])


if __name__ == "__main__":
    PATHS_LABEL = [Path("../../../data/2_processed_positions/tries/vid0.csv"),
                   Path("../../../data/2_processed_positions/tries/vid1.csv")]
    try:
        GENERATOR = generate_data(PATHS_LABEL, take_all=False)
        print(GENERATOR)
    except FindPathDataError as find_path_data_error:
        print(find_path_data_error.__repr__())
