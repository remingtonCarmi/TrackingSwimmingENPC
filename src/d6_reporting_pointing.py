"""
This module compares the pointed lane between two users.
"""
import numpy as np
import cv2


def compute_common_indexes(pointing1, pointing2):
    """
    Computes the common pointed indexes.

    Args:
        pointing1 (DataFrame): the pointed lanes from a user.

        pointing2 (DataFrame): the pointed lanes from another user.

    Returns:
        pointed_index (Index): the common pointed indexes.
    """
    # Select the lines where both lanes have been pointed
    pointed_index1 = pointing1.index[pointing1["x_head"] >= 0]
    pointed_index2 = pointing2.index[pointing2["x_head"] >= 0]

    pointed_index = pointed_index2.intersection(pointed_index1).sort_values()

    return pointed_index


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    # To get the distance in the video
    from src.d7_visualization.tools.get_meters_video import get_meters_video

    # Define the paths
    VIDEO_NAME = "vid1"
    PATH_VIDEO = Path("../data/1_raw_videos/{}.mp4".format(VIDEO_NAME))
    PATH_POINTING_REMI = Path("../data/3_processed_positions/tries/{}_pointing_remi.csv".format(VIDEO_NAME))
    PATH_POINTING_THEO = Path("../data/3_processed_positions/tries/{}_pointing_theo.csv".format(VIDEO_NAME))

    PATH_CALIBRATION = Path("../data/2_intermediate_top_down_lanes/calibration/{}.txt".format(VIDEO_NAME))

    # Get the distance in the video
    (left_limit, video_lenght_meter) = get_meters_video(PATH_CALIBRATION)
    video_capture = cv2.VideoCapture(str(PATH_VIDEO))
    video_lenght_pixel = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Load the csv files
    remi_pointing = pd.read_csv(PATH_POINTING_REMI)
    theo_pointing = pd.read_csv(PATH_POINTING_THEO)

    # Compute the common indexes
    common_pixels = compute_common_indexes(remi_pointing, theo_pointing)

    # Compute the distance
    distance_l1 = np.linalg.norm(remi_pointing.loc[common_pixels] - theo_pointing.loc[common_pixels], ord=1) / len(common_pixels)
    print("The average distance is :", distance_l1, "pixels")
    print("The average distance is :", distance_l1 * video_lenght_meter / video_lenght_pixel, "meters")

    # Percentage of common pixels
    print("The percentage of common pointed lanes is", 100 * len(common_pixels) / len(remi_pointing))

    # Percentage of pointed pixels
    print("The percentage of pointed lanes for Rémi is", 100 * np.sum(remi_pointing["x_head"] >= 0) / len(remi_pointing))
    print("The percentage of pointed lanes for Théo is", 100 * np.sum(theo_pointing["x_head"] >= 0) / len(theo_pointing))