"""
This file makes a video with a list of images.
"""
from pathlib import Path
import cv2
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, FindErrorStore


def make_video(name_video, images, fps=25, destination=Path("../output/test/")):
    """
    Makes a video with all the images in images.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the images.

        fps (int): the fps of the created video.

        destination (pathlib): the path leading to the folder where the video will be registered.
    """
    # Check that the folder exists
    if not destination.exists():
        raise FindErrorStore(destination)

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
