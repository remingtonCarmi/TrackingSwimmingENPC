"""
Add blank information to a csv file.
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import cv2


def create_csv(csv_name, destination=None, dictionary=None):
    """
    Create a csv file.

    Args:
        csv_name (string): the name of the csv file.

        destination (WindowsPath): the path where the csv will be created.
    """
    if destination is None:
        destination = Path("../../../data/3_processed_positions/tries")

    if dictionary is None:
        dictionary = {"x_head": [], "y_head": []}

    csv_path = destination / csv_name

    # Create the keys
    keys = pd.DataFrame(dictionary)

    # saving the dataframe
    keys.to_csv(csv_path, index=False)


def fill_csv_with_blank(path_csv, idx_lane, frame_begin, frame_end):
    """
    Fill the csv with blank information for a lane.
    """
    # Get the paths
    video_name = path_csv.parts[-1][:-4]
    destination_csv = path_csv.parent
    csv_name_save = "{}.csv".format(video_name)
    csv_path_save = destination_csv / csv_name_save

    data_frame = pd.read_csv(path_csv)

    # Set the new indexes
    new_index = [(idx_lane, idx_frame) for idx_frame in range(frame_begin, frame_end + 1)]
    mux = pd.MultiIndex.from_tuples(new_index)

    # Set the new data
    new_data = {"x_head": -1.0, "y_head": -1.0, "swimming_way": 1.0}

    new_data_frame = pd.DataFrame(new_data, index=mux)

    filled_data_frame = pd.concat([data_frame.loc[:idx_lane], new_data_frame, data_frame.loc[idx_lane:]])

    # Create the new csv
    create_csv(csv_name_save, destination=destination_csv, dictionary={"x_head": [], "y_head": [], "swimming_way": []})

    with open(csv_path_save, "a", newline="") as csv_file:
        # Save the changes
        filled_data_frame.to_csv(csv_file, header=False)


if __name__ == "__main__":
    PATH_CSV = Path("data/3_processed_positions/full_100_NL_D_FA-Canet.csv")

    fill_csv_with_blank(PATH_CSV, 8, 96, 1295)
