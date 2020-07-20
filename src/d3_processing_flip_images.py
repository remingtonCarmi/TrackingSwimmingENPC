"""
Add the flip label to the csv files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Exceptions
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError

# To create a cvs file
from src.d0_utils.store_load_data.fill_csv import create_csv


def get_ways(indexes, changes):
    """
    Get the flip label.

    Args:
        indexes (MultiIndex): the indexes of the video.

        changes (array): the index where a change has to be made.

    Returns:
        swimming_ways (array): a list that indicate on which way the swimmer is swimming.
            + 1 if the swimmer goes toward the left.
            - 1 if the swimmer goes toward the right.
    """
    indexes_change = pd.MultiIndex.from_tuples(tuple(changes))
    indexes_search = indexes.searchsorted(indexes_change)
    swimming_ways = np.ones(len(indexes))

    nb_changes = len(changes)
    for idx_index in range(0, nb_changes, 2):
        if idx_index + 1 < nb_changes:
            swimming_ways[indexes_search[idx_index]: indexes_search[idx_index + 1]] = -1
        else:
            swimming_ways[indexes_search[idx_index]:] = -1

    return swimming_ways


def add_swimming_way(video_name, destination_csv, changes):
    """
    Add a column that indicates the swimming way of the swimmers.

    Args:
        video_name (string): the name of the video

        destination_csv (WindowsPath): the path that lead to the csv path.

        changes (list of list of integers): the list of the indexes where swimmers change direction.
    """
    # Get the paths
    csv_path = destination_csv / "{}.csv".format(video_name)
    csv_name_save = "full_{}.csv".format(video_name)
    csv_path_save = destination_csv / csv_name_save

    if not csv_path.exists():
        raise FindPathError(csv_path)

    # Create the new csv
    create_csv(csv_name_save, destination=destination_csv, dictionary={'x_head': [], 'y_head': [], 'swimming_way': []})

    # Load the data
    data_frame = pd.read_csv(csv_path)

    with open(csv_path_save, 'a', newline='') as csv_file:
        # Add the swimming_way column to the data frame
        data_frame["swimming_way"] = get_ways(data_frame.index, changes)

        # Save the changes
        data_frame.to_csv(csv_file, header=False)


if __name__ == "__main__":
    VIDEO_NAME = "100_NL_F_FA"
    DESTINATION_CSV = Path("../data/3_processed_positions/tries")

    CHANGES = np.array([[1, 0], [1, 960], [2, 0], [2, 960], [3, 0], [3, 960], [4, 0], [4, 960], [5, 0], [5, 960], [6, 0], [6, 960], [7, 0], [7, 960], [8, 0], [8, 960]])

    try:
        add_swimming_way(VIDEO_NAME, DESTINATION_CSV, CHANGES)
    except FindPathError as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
