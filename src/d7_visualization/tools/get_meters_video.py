"""
This module computes the left limit in meters of a video.
"""
import numpy as np


def get_meters_video(path_calibration):
    """
    Get the left limit and the length of video in meters.

    Args:
        path_calibration (WindowsPath): the path that leads to the calibration file of the video.

    Returns:
        (float): the left limit of the video in meters.

        (float): the length of the video in meters.
    """
    file = open(path_calibration, 'r')
    lines = file.readlines()
    extreme_values = np.fromstring(lines[-1], dtype=float, sep=',')

    return extreme_values[0], abs(extreme_values[2] - extreme_values[0])
