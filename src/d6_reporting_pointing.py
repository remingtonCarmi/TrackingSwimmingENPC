"""
This module compares the pointed lane between two users.
"""
import numpy as np
import cv2


def compute_common_indexes(pointing1, pointing2, pointing3, three_pointer):
    """
    Computes the common pointed indexes.

    Args:
        pointing1 (DataFrame): the pointed lanes from a user.

        pointing2 (DataFrame): the pointed lanes from another user.

        pointing3 (DataFrame): the pointed lanes from another user.

        three_pointer (boolean): indicates if the third pointer has to be taken into account.

    Returns:
        pointed_index (Index): the common pointed indexes.
    """
    # Select the lines where both lanes have been pointed
    pointed_index1 = pointing1.index[pointing1["x_head"] >= 0]
    pointed_index2 = pointing2.index[pointing2["x_head"] >= 0]

    pointed_index = pointed_index2.intersection(pointed_index1).sort_values()

    if three_pointer:
        # Select the lines where the lanes have been pointed
        pointed_index3 = pointing3.index[pointing3["x_head"] >= 0]

        pointed_index = pointed_index3.intersection(pointed_index).sort_values()

    return pointed_index


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    # To get the distance in the video
    from src.d7_visualization.tools.get_meters_video import get_meters_video

    # Variables
    POINTER_1 = "remi"
    POINTER_2 = "clem"
    POINTER_3 = "theo"
    THREE_POINTER = True

    # Define the paths
    VIDEO_NAME = "vid1"
    PATH_VIDEO = Path("../data/1_raw_videos/{}.mp4".format(VIDEO_NAME))
    PATH_POINTING_1 = Path("../data/3_processed_positions/{}_pointing_{}.csv".format(VIDEO_NAME, POINTER_1))
    PATH_POINTING_2 = Path("../data/3_processed_positions/{}_pointing_{}.csv".format(VIDEO_NAME, POINTER_2))
    PATH_POINTING_3 = Path("../data/3_processed_positions/{}_pointing_{}.csv".format(VIDEO_NAME, POINTER_3))
    PATH_CALIBRATION = Path("../data/2_intermediate_top_down_lanes/calibration/{}.txt".format(VIDEO_NAME))

    # Get the distance in the video
    (left_limit, video_length_meter) = get_meters_video(PATH_CALIBRATION)
    video_capture = cv2.VideoCapture(str(PATH_VIDEO))
    video_lenght_pixel = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Load the csv files
    pointing1 = pd.read_csv(PATH_POINTING_1)
    pointing2 = pd.read_csv(PATH_POINTING_2)
    pointing3 = pd.read_csv(PATH_POINTING_3)

    # Compute the common indexes
    common_pixels = compute_common_indexes(pointing1, pointing2, pointing3, THREE_POINTER)

    # Compute the distance
    if THREE_POINTER:
        pointings = np.zeros((len(common_pixels), 3))
        pointings[:, 0] = pointing1.loc[common_pixels]["x_head"]
        pointings[:, 1] = pointing2.loc[common_pixels]["x_head"]
        pointings[:, 2] = pointing3.loc[common_pixels]["x_head"]

        max_pointings = np.max(pointings, axis=1)
        min_pointings = np.min(pointings, axis=1)

        distance_l1 = np.linalg.norm(max_pointings - min_pointings, ord=1) / len(common_pixels)

    else:
        distance_l1 = np.linalg.norm(pointing1.loc[common_pixels]["x_head"] - pointing2.loc[common_pixels]["x_head"],
                                     ord=1) / len(common_pixels)

    print("The average distance is :", distance_l1, "pixels")
    print("The average distance is :", distance_l1 * video_length_meter / video_lenght_pixel, "meters")

    # Percentage of common pixels
    print("The percentage of common pointed lanes is", 100 * len(common_pixels) / len(pointing1))

    # Percentage of pointed pixels
    print("The percentage of pointed lanes for {} is".format(POINTER_1),
          100 * np.sum(pointing1["x_head"] >= 0) / len(pointing1))
    print("The percentage of pointed lanes for {} is".format(POINTER_2),
          100 * np.sum(pointing2["x_head"] >= 0) / len(pointing2))
    if THREE_POINTER:
        print("The percentage of pointed lanes for {} is".format(POINTER_3),
              100 * np.sum(pointing3["x_head"] >= 0) / len(pointing3))
