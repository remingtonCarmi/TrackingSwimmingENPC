"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import cv2
import numpy as np
import random as rd
from src.utils.extract_image import extract_image_video
from src.utils.extract_image import TimeError
# from src.perspective.correction_perspective import correct_perspective_img
from src.utils.exception_classes import VideoFindError
from src.utils.point_selection.point_selection import perspective_selection
import matplotlib.pyplot as plt


def make_video(name_video, images, fps=25, destination=Path("../output/test/")):
    """
    Makes a video with all the images in images.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the images.
    """
    height, width, layers = images[0].shape
    path_video = destination / name_video
    size = (width, height)
    out = cv2.VideoWriter(str(path_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for image in images:
        out.write(image)
    out.release()


def correct_perspective_img(image, src, dst):
    # we find the transform matrix M thanks to the matching of the four points
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    # warp the image to a top-down view
    warped = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped


def transform_in_2d(points, h_dim, w_dim):
    points[:, 0] = (points[:, 0] + 1) * w_dim / 52
    points[:, 1] = (points[:, 1] + 1) * h_dim / 27


def calibrate_video(name_video, time_begin=0, time_end=-1, destination_video=Path("../output/test/")):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        name_video (string): name of the video.

        time_begin (integer): the starting time in second. Default value = 0.

        time_end (integer): the ending time in second. Default value = -1.

        destination_video (string): the destination path of the cleaned video
            Default value = "".
    """
    # Get the images
    print("Get the images ...")
    list_images = extract_image_video(name_video, time_begin, time_end)
    nb_images = len(list_images)
    (height, width) = list_images[0].shape[: 2]

    # Selection of the perspective points on a random image
    print("Point selection ...")
    (points_image, points_real) = perspective_selection(list_images[rd.randint(int(nb_images / 2), nb_images - 1)])
    transform_in_2d(points_real, height, width)

    # Transform the images
    print("Correction of images ...")
    for index_image in range(nb_images):
        list_images[index_image] = correct_perspective_img(list_images[index_image], points_image, points_real)

    # Get the fps
    video = cv2.VideoCapture(str(name_video))
    fps_video = int(video.get(cv2.CAP_PROP_FPS))

    print("Make the corrected video ...")
    make_video("test1.mp4", list_images, fps_video, destination_video)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid0.mp4")
    DESTINATION_VIDEO = Path("../data/videos/corrected/")
    try:
        calibrate_video(PATH_VIDEO, 0, 1, DESTINATION_VIDEO)
    except TimeError as time_error:
        print(time_error.__repr__())
    except VideoFindError as video_find_error:
        print(video_find_error.__repr__())
