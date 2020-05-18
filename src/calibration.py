"""
This code calibrates an entire video by withdrawing the perspectives.

It can create a video that with a top-down view.
It can create a txt file to register the parameters of the calibration.
"""
from pathlib import Path
import random as rd
import numpy as np
import cv2
from src.utils.extractions.extract_image import extract_image_video
from src.utils.extractions.exceptions.exception_classes import TimeError
from src.utils.extractions.exceptions.exception_classes import FindErrorExtraction
from src.utils.make_video import make_video
from src.utils.point_selection.calibration_selection import calibration_selection
from src.utils.perspective_correction.perspective_correction import get_top_down_image, get_homography
from src.utils.store_load_matrix.fill_txt import store_calibration_txt
from src.utils.store_load_matrix.exception_classes import AlreadyExistError


def meter_to_pixel(src_points, dst_meter, image):
    """
    Transforms the points that where given in meters to point in pixels so that
    the calibrated image is the biggest without loosing information.

    Args:
        src_points (array, shape = (4, 2)): the points in pixel selected in the original image.

        dst_meter (array, shape = (4, 2)): the points in meter that corresponds to the src_points.

        image (array, shape = the original shape of the video): the image from which the points
            where taken.

    Returns:
        dst_pixel_full_pool (array, shape = (4, 2)): the points in pixel that corresponds to the src_points
            to show the part of the pool, on which we have information, from the top.

        extreme_points (list of lists of 2 elements): the 2 extreme points in meter.
    """
    # Get the image size
    (height, width) = image.shape[: 2]

    # --- Get the top down view of the entire pool --- #
    # Get the coordinates in pixel of dst_meter in the entire pull
    dst_pixel_full_pool = np.zeros((4, 2))
    # We take one meter from each side to be sure that we do not lose information
    # Hence : + 1 and 52 meters instead of 50 meters.
    dst_pixel_full_pool[:, 0] = (dst_meter[:, 0] + 1) * width / 52
    dst_pixel_full_pool[:, 1] = dst_meter[:, 1] * height / 25

    # Transformation of the original image
    homography = get_homography(src_points, dst_pixel_full_pool)
    top_down_image = get_top_down_image(image, homography)

    # --- Get the top down view of the pool that we can see --- #
    # Find the first column that is not black
    index_w = 0
    while index_w < width and np.sum(top_down_image[:, index_w]) == 0:
        index_w += 1
    left_column = index_w

    # Find the last column that is not black
    index_w = width - 1
    while index_w >= 0 and np.sum(top_down_image[:, index_w]) == 0:
        index_w -= 1
    right_column = index_w

    # Compute the extreme points
    # We add -1 since the top left point of the top down view of the full pool is [-1, 0]
    top_left = [left_column * 52 / width - 1, 0]
    bottom_right = [right_column * 52 / width - 1, 25]

    # Get the coordinates in pixel of dst_pixel_full_pool in the top down view of the pool that we can see
    dst_pixel_full_pool[:, 0] = (dst_pixel_full_pool[:, 0] - left_column) / (right_column - left_column) * width

    return dst_pixel_full_pool, [top_left, bottom_right]


def calibrate_video(path_video, time_begin=0, time_end=-1, destination_video=Path("../output/test/"),
                    destination_txt=Path("../output/test"), create_video=False, create_txt=False):
    """
    Calibrates the video from the starting time to the end time.

    Args:
        path_video (pathlib): the path where the video is. Should lead to the video file.

        time_begin (integer): the starting time in second.
            Default value = 0.

        time_end (integer): the ending time in second. If -1, the calibration is done until the end.
            Default value = -1.

        destination_video (pathlib): the destination path of the calibrated video. Should lead to a folder.
            Default value = Path("../output/test/").

        destination_txt (pathlib): the destination path of the txt file that explains how the video can be calibrated.
            Default value = Path("../output/test/").

        create_video (boolean): if True, create the video in the destination video path.
            Default value = False.

        create_txt (boolean): if True, create a txt file that tells how to calibrate the video.
            Default value = True.

    Returns:
        list_images (array, shape = (number of image in the original video, shape of an image in the
            original video): the list of the calibrated images.

    If the video does not exist, an FindErrorExtraction will be raised.
    If the beginning time or the ending time are not well defined, an TimeError will be raised.
    If the txt file is already created, an AlreadyExistError exception will be raised.
    """
    # Verify that the folders exist and that the video or the txt file does not exist
    name_video = path_video.parts[-1]
    if create_video:
        # Check that the folder exists
        if not destination_video.exists():
            raise FindErrorExtraction(destination_video)

        corrected_video = "corrected_" + name_video
        path_corrected_video = destination_video / corrected_video
        # Check that the video does not exist
        if path_corrected_video.exists():
            raise AlreadyExistError(path_corrected_video)
    if create_txt:
        # Check that the folder exists
        if not destination_txt.exists():
            raise FindErrorExtraction(destination_txt)

        name_txt = name_video[: -3] + "txt"
        path_txt = destination_txt / name_txt
        # Check that the video does not exist
        if path_txt.exists():
            raise AlreadyExistError(path_txt)

    # Get the images
    print("Get the images ...")
    list_images = extract_image_video(path_video, time_begin, time_end)
    nb_images = len(list_images)

    # Selection of the 8 points in a random image
    print("Point selection for calibration ...")
    (points_src, points_meter) = calibration_selection(list_images[rd.randint(int(nb_images / 10), int(nb_images / 5))])

    # Get the coordinate of the points in the final image in pixels and the extreme points
    (points_dst, exteme_points) = meter_to_pixel(points_src, points_meter, list_images[0])

    # Get the homography matrix
    homography = get_homography(points_src, points_dst)

    # Transform the images
    print("Correction of images ...")
    for index_image in range(nb_images):
        list_images[index_image] = get_top_down_image(list_images[index_image], homography)

    if create_video:
        # Get the fps
        video = cv2.VideoCapture(str(path_video))
        fps_video = int(video.get(cv2.CAP_PROP_FPS))

        # Make the video
        print("Make the corrected video ...")
        make_video(corrected_video, list_images, fps_video, destination_video)

    if create_txt:
        # Construct the txt file
        to_store = [name_video, points_src, points_dst, homography, exteme_points]
        store_calibration_txt(name_txt, to_store, destination_txt)

    return np.array(list_images)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid0.mp4")
    try:
        calibrate_video(PATH_VIDEO, 10, 11, create_video=True, create_txt=True)
    except FindErrorExtraction as video_find_error:
        print(video_find_error.__repr__())
    except TimeError as time_error:
        print(time_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
