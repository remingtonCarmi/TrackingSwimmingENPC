"""
This module fills a csv file for the pointing.
It also enables to find the last line of a csv file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError, NothingToAddError


def last_line(csv_path):
    """
    Find the last line of the csv file.

    Args:
        csv_path (WindowsPath): the path that leads to the csv file.

    Returns:
        lane (integer): the index of the lane.
            lane = -1 if the csv file is empty.

        frame (integer): the index of the frame.
            frame = -1 if the csv file is empty.
    """
    # Verify that the csv file exists
    if not csv_path.exists():
        raise FindPathError(csv_path)

    # Get the last line
    data_frame = pd.read_csv(csv_path)

    if not data_frame.empty:
        (lane, frame) = data_frame.index.values[-1]
    else:
        lane = -1
        frame = -1

    return lane, frame


def create_csv(csv_name, destination=None):
    """
    Create a csv file.

    Args:
        csv_name (string): the name of the csv file.

        destination (WindowsPath): the path where the csv will be created.
    """
    if destination is None:
        destination = Path("../../../data/2_processed_positions/tries")

    # Check that the folder exists
    if not destination.exists():
        raise FindPathError(destination)

    csv_path = destination / csv_name
    if csv_path.exists():
        raise AlreadyExistError(csv_path)

    # Create the keys
    dictionary = {'x_head': [], 'y_head': []}
    keys = pd.DataFrame(dictionary)

    # saving the dataframe
    keys.to_csv(csv_path, index=False)


def fill_csv(csv_path, lanes_frames, points):
    """
    Fill the csv with the given points.

    Args:
        csv_path (WindowsPath): the path that leads to the csv path.

        lanes_frames (array): the list of the indexes of the pointed images.

        points (array): the list of the points linked with the pointed images.
    """
    # Check that the folder exists
    if not csv_path.exists():
        raise FindPathError(csv_path)

    if len(lanes_frames) == 0:
        raise NothingToAddError(csv_path)

    with open(csv_path, 'a', newline='') as csv_file:
        # Create lines
        dictionary = {'x_head': points[:, 0], 'y_head': points[:, 1]}

        # Create the indexes
        index = pd.MultiIndex.from_tuples(tuple(lanes_frames))

        # Create data frame
        new_points = pd.DataFrame(dictionary, index=index)

        # Add data frame
        new_points.to_csv(csv_file, header=False)


if __name__ == "__main__":
    CVS_NAME = "test0.csv"
    CVS_PATH = Path("../../../data/2_processed_positions/tries") / CVS_NAME
    LIST_LANES_FRAMES = np.array([[1, 5], [4, 89]])
    LIST_POINTS = np.array([[4, 4.9], [9, 2.9]])

    try:
        # create_csv(CVS_NAME)
        fill_csv(CVS_PATH, LIST_LANES_FRAMES, LIST_POINTS)
        LAST_LINE = last_line(CVS_PATH)
        print(LAST_LINE)
    except FindPathError as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
    except NothingToAddError as nothing_to_add:
        print(nothing_to_add.__repr__())
