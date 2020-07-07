"""
This file makes a video with a list of image.
"""
from pathlib import Path
import cv2
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError


def make_video(name_video, images, fps=25, destination=None):
    """
    Makes a video with all the image in image.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the image.

        fps (int): the fps of the created video.

        destination (pathlib): the path leading to the folder where the video will be registered.
    """
    if destination is None:
        destination = Path("../../data/4_model_output/tries/videos")

    # Check that the folder exists
    if not destination.exists():
        raise FindPathError(destination)

    # Verify that the video does not exist
    path_video = destination / name_video
    if path_video.exists():
        raise AlreadyExistError(path_video)

    # Parameters
    (height, width) = images[0].shape[: 2]
    size = (width, height)

    out = cv2.VideoWriter(str(path_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for image in images:
        out.write(image)
    out.release()
