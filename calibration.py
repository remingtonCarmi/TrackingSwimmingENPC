"""
This code calibrate an entire video by withdrawing the distortion and the perspectives.
"""
import cv2
import numpy as np
import glob
from extract_image import extract_image_video

def make_video(name_video, images):
    """
    Makes a video with all the images in images.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the images.
    """
    size = images[0].shape
    out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for image in images:
        out.write(image)
    out.release()


def calibrate_video(name_video, time_begin, time_end):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        name_video (string): name of the video.

        time_begin (integer): the starting time in second.

        time_end (integer): the ending time in second.
    """


if __name__ == "__main__":
    images = [extract_image_video()]
    make_video()