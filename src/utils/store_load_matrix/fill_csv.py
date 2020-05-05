from pathlib import Path
import csv
import numpy as np
import pandas as pd
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, FindErrorStore, NothingToAdd


def last_line(csv_path):
    # Verify that the csv file exists
    if not csv_path.exists():
        raise FindErrorStore(csv_path)

    # Get the last line
    data_frame = pd.read_csv(csv_path)

    if not data_frame.empty:
        (frame, lane) = data_frame.index.values[-1]
    else:
        frame = -1
        lane = -1
    return frame, lane


def create_csv(csv_name, destination_path=Path("../../../output/test/")):
    # Check that the folder exists
    if not destination_path.exists():
        raise FindErrorStore(destination_path)

    csv_path = destination_path / csv_name
    if csv_path.exists():
        raise AlreadyExistError(csv_path)

    # Create the keys
    dictionary = {'x_head': [], 'y_head': []}
    keys = pd.DataFrame(dictionary)

    # saving the dataframe
    keys.to_csv(csv_path, index=False)


def fill_csv(csv_path, frame_lane, points):
    # Check that the folder exists
    if not csv_path.exists():
        raise FindErrorStore(csv_path)

    if len(frame_lane) == 0:
        raise NothingToAdd(csv_path)

    with open(csv_path, 'a', newline='') as csv_file:
        # Create lines
        dictionary = {'x_head': points[:, 0], 'y_head': points[:, 1]}

        # Create the indexes
        index = pd.MultiIndex.from_tuples(tuple(frame_lane))

        # Create data frame
        new_points = pd.DataFrame(dictionary, index=index)

        # Add data frame
        new_points.to_csv(csv_file, header=False)


if __name__ == "__main__":
    CVS_NAME = "test0.csv"
    CVS_PATH = Path("../../../output/test/") / CVS_NAME
    LIST_IMAGE_NAME = np.array([[0, 5], [34, 8]])
    LIST_POINTS = np.array([[4, 4.9], [9, 2.9]])
    try:
        # create_csv(CVS_NAME)
        fill_csv(CVS_PATH, LIST_IMAGE_NAME, LIST_POINTS)
        LAST_LINE = last_line(CVS_PATH)
        print(LAST_LINE)
    except FindErrorStore as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
    except NothingToAdd as nothing_to_add:
        print(nothing_to_add.__repr__())
