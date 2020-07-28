"""
This module undo the perspective transformation that has been done to a bird-view camera.
"""
from pathlib import Path
import numpy as np
import cv2


def read_homography(path_calibration):
    """
    Takes the homography from the calibration file.

    Args:
        path_calibration (WindowsPath): path that leads to the calibration file.

    Returns:
        (array of shape : (3, 3)): the homography matrix.
    """
    # Get the homography
    file = open(path_calibration, 'r')
    lines = file.readlines()
    homography = np.fromstring(lines[-2], dtype=float, sep=',')
    file.close()

    return np.reshape(homography, (3, 3))


def get_original_image(transformed_image, homography, original_dimensions):
    """
    Get the original image.

    Args:
        transformed_image (array): the transformed image.

        homography (array of shape : (3, 3)) : the matrix of the homography.

        original_dimensions (array of integer): the dimensions of the original video.

    Returns:
        (array): the original image
    """
    # warp the image to the original image
    return cv2.warpPerspective(transformed_image, np.linalg.inv(homography), (original_dimensions[1], original_dimensions[0]), flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    # Define the paths
    CALIBRATION_PATH = Path("../../../data/2_intermediate_top_down_lanes/calibration/tries/vid0.txt")
    PATH_ORIGINAL_IMAGE = Path("../../../data/5_model_output/tries/raw_images/vid0_frame458.jpg")

    # Parameters
    DIMENSIONS = np.array([108, 1920])
    ORIGINAL_DIMENSIONS = np.array([1080, 1920])
    LABEL = np.array([56, 1314])

    # Load the image
    ORIGINAL_IMAGE = cv2.imread(str(PATH_ORIGINAL_IMAGE)).astype(float)

    # Get the homography
    HOMOGRAPHY = read_homography(CALIBRATION_PATH)

    # Get the prediction
    PRED_IMAGE = np.zeros((ORIGINAL_DIMENSIONS[0], DIMENSIONS[1], 3))
    PRED_IMAGE[int(6 * ORIGINAL_DIMENSIONS[0] / 10): int(7 * ORIGINAL_DIMENSIONS[0] / 10), LABEL[1]] = [-255, -255, 255]
    PRED_IN_ORIGINAL = get_original_image(PRED_IMAGE, HOMOGRAPHY, ORIGINAL_DIMENSIONS)

    # Add the prediction to the original image
    ORIGINAL_IMAGE += PRED_IN_ORIGINAL
    np.clip(ORIGINAL_IMAGE, 0, 255, ORIGINAL_IMAGE)

    cv2.imshow("Original image", ORIGINAL_IMAGE.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
