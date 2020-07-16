"""
This module standardizes the input image.
"""
import numpy as np
from pathlib import Path
import cv2


def standardize(image):
    """
    Standardizes the image.

    Args:
        image (array): the input image.

    Returns:
        (array): the standardized image.
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../../data/4_model_output/tries/transformed_images/rescaled_l1_f0107.jpg")

    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    LABEL = np.array([83, 2903])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE))

    # Standardize the image
    STAN_IMAGE = standardize(IMAGE)

    # Plot original image
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    cv2.imshow("Image", IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plot standardized image
    STAN_IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    # cv2.imwrite(str(PATH_SAVE), STAN_IMAGE)
    cv2.imshow("Standardized image", STAN_IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
