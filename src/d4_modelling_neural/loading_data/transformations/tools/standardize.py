"""
This module standardizes the input image.
"""
import numpy as np
from pathlib import Path
import cv2


def standardize(image):
    """
    Standardizes the image.
    In reality, it withdraws 100 with divides by 50.

    Args:
        image (array): the input image.
    """
    image -= 100.
    image /= 50.


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/100NL_FAF/l8_f1054.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../../data/5_model_output/tries/transformed_images/rescaled_l1_f0107.jpg")

    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    # LABEL = np.array([83, 2903])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE)).astype(np.float)

    # Plot original image
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    cv2.imshow("Image", IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Standardize the image
    standardize(IMAGE)

    # Plot standardized image
    print(IMAGE)
    print("Mean", np.mean(IMAGE))
    print("Std", np.std(IMAGE))
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    # cv2.imwrite(str(PATH_SAVE), STAN_IMAGE)
    cv2.imshow("Standardized image", IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
