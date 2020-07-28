"""
This module serves to calibrate image of a video thanks to a txt file.
"""
from pathlib import Path
import numpy as np

# Exceptions
from src.d0_utils.extractions.exceptions.exception_classes import FindPathExtractError

# To get the homography
from src.d0_utils.perspective_correction.undo_perspective import read_homography

# To extract image from a video
from src.d0_utils.extractions.extract_image import extract_image_video

# To transform image to get the top-down view
from src.d0_utils.perspective_correction.perspective_correction import get_top_down_image


def calibrate_from_txt(path_video, path_txt, time_begin=0, time_end=-1):
    """
    Extract the image and transform them to get the top-down view.

    Args:
        path_video (WindowsPath): the path that leads to the video.

        path_txt (WindowsPath): the path that leads to the txt file for the calibration.

        time_begin (integer): the beginning time in second.
            Default value = 0

        time_end (integer): the ending time in second.
            Default value = -1
            if time_end == -1, the video is viewed until the end.

    Returns:
        list_images (list of array): list of the calibrated image.
    """
    # Check that the paths exists
    if not path_video.exists():
        raise FindPathExtractError(path_video)
    if not path_txt.exists():
        raise FindPathExtractError(path_txt)

    # Get the homography
    homography = read_homography(path_txt)

    # Get the image
    list_images = extract_image_video(path_video, time_begin, time_end)
    nb_images = len(list_images)

    # Transform the image
    for index_image in range(nb_images):
        list_images[index_image] = get_top_down_image(list_images[index_image], homography)

    return list_images


if __name__ == "__main__":
    import cv2
    # 100_NL_F_FA / l5_f0123
    PATH_VIDEO = Path("../../data/1_raw_videos/100_NL_F_FA.mp4")
    PATH_TXT = Path("../../data/2_intermediate_top_down_lanes/calibration/100_NL_F_FA.txt")

    PATH_SAVE = Path("../../data/5_model_output/tries/top_down_images")

    IMAGES = calibrate_from_txt(PATH_VIDEO, PATH_TXT, 5, 6)
    print(IMAGES[0].shape)
    cv2.imshow("Image", IMAGES[3].astype(np.uint8))
    cv2.imwrite(str(PATH_SAVE / "top_down_l5_f0123.jpg"), IMAGES[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
