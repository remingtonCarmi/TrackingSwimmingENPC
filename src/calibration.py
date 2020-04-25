"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import cv2
import numpy as np
import random as rd
from src.utils.extract_image import extract_image_video
from src.utils.extract_image import TimeError
from src.utils.exception_classes import VideoFindError
from src.utils.make_video import make_video
from src.utils.point_selection.calibration_selection import calibration_selection
from src.utils.perspective_correction.perspective_correction import get_top_down_image, get_homography
from src.utils.store_load_matrix.start_csv import store_calibration_csv, CSVExistError


def meter_to_pixel(src_points, dst_meter, image):
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
                    create_video=True, creat_csv=False):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        path_video (string): name of the video.

        time_begin (integer): the starting time in second. Default value = 0.

        time_end (integer): the ending time in second. Default value = -1.

        destination_video (string): the destination path of the cleaned video
            Default value = "".
    """
    # Get the images
    print("Get the images ...")
    list_images = extract_image_video(path_video, time_begin, time_end)
    nb_images = len(list_images)

    # Selection of the 8 points in a random image
    print("Point selection ...")
    (points_src, points_meter) = calibration_selection(list_images[rd.randint(int(nb_images / 10), int(nb_images / 5))])

    # Get the coordinate of the points in the final image in pixels and the extreme points
    (points_dst, exteme_points) = meter_to_pixel(points_src, points_meter, list_images[0])

    # Get the homography matrix
    homography = get_homography(points_src, points_dst)

    # Transform the images
    print("Correction of images ...")
    for index_image in range(nb_images):
        list_images[index_image] = get_top_down_image(list_images[index_image], homography)

    name_video = path_video.parts[-1]
    if create_video:
        # Get the fps
        video = cv2.VideoCapture(str(path_video))
        fps_video = int(video.get(cv2.CAP_PROP_FPS))

        # Make the video
        print("Make the corrected video ...")
        corrected_video = "corrected_" + name_video
        make_video(corrected_video, list_images, fps_video, destination_video)

    if creat_csv:
        # Construct the csv file
        to_store = [name_video, points_dst, points_dst, homography, exteme_points]
        store_calibration_csv(name_video[: -3] + "csv", to_store, destination_video)

    return np.array(list_images)

    
if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid1.mp4")
    DESTINATION_VIDEO = Path("../data/videos/corrected/")
    try:
        calibrate_video(PATH_VIDEO, 10, 11, DESTINATION_VIDEO)
    except TimeError as time_error:
        print(time_error.__repr__())
    except VideoFindError as video_find_error:
        print(video_find_error.__repr__())
    except CSVExistError as exist_error:
        print(exist_error.__repr__())
