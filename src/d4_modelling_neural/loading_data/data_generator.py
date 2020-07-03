"""
This module generate an object that holds the paths to the images
and the labels of these images.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def generate_data(paths_data, paths_label, take_all=False):
    """
    Produce a list of list of 3 elements : path_to_image, x_head, y_head.
    We look in the labels and if the image is in the computer, we add it to the list.

    Args:
        paths_data (WindowsPath): the paths to the images.

        paths_label (WindowsPath): the paths to the labels.

        take_all (boolean): if True, all the images are taken. Otherwise, only the images
            with a positive label is taken.

    Returns:
        full_data (list of list of 3 elements : (WindowsPath, integer, integer):
            list of elements like (path_to_image, x_head, y_head) for the training
    """
    # !! EXCEPTIONS : Paths exist, len(DATA) = len(LABELS), len(DATA) > 0 !!
    nb_videos = len(paths_data)
    full_data = []

    for idx_video in range(nb_videos):
        # Get the labels
        labels = pd.read_csv(paths_label[idx_video])

        # Get the data
        data = get_full_data(paths_data[idx_video], labels, take_all)

        # Add the data
        full_data.extend(data)

    return full_data


def get_full_data(path_data, labels, take_all):
    """
    Produce a list of list of 3 elements : path_to_image, x_head, y_head.
    We look in the labels and if the image is in the computer, we add it to the list.

    Args:
        path_data (WindowsPath): path to the images.

        labels (DataFrame): table that contains the labels.

        take_all (boolean): if True, all the images are taken. Otherwise, only the images
            with a positive label is taken.

    Returns:
        full_data (list of list of 3 elements : (WindowsPath, integer, integer):
            list of elements like (path_to_image, x_head, y_head) for the training
    """
    full_data = []
    for ((lane, frame), label) in labels.iterrows():
        # Add to the list only if the image has been labeled with a right position
        if label[0] >= 0 or take_all:
            name_image = "l{}_f{}.jpg".format(lane, str(frame).zfill(4))
            path_image = path_data / name_image
            # Add to the list only if the image is in the computer
            if path_image.exists():
                full_data.append([path_image, label[0], label[1]])

    return full_data


if __name__ == "__main__":
    PATHS_DATA = [Path("../../../data/1_intermediate_top_down_lanes/lanes/tries/vid0")]
    PATHS_LABEL = [Path("../../../data/2_processed_positions/tries/vid0.csv")]

    GENERATOR = generate_data(PATHS_DATA, PATHS_LABEL, take_all=True)
    print(GENERATOR)
