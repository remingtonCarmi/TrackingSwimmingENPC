"""
This module flips image around the vertical axis.
"""
from pathlib import Path
import numpy as np
import cv2


def flip_image(image, label):
    """
    Flip the image around the vertical axis.

    Args:
        image (3d array): the image.

        label (1d array of float): [y_head, x_head].
    """
    # Flip the image
    cv2.flip(image, 1, image)

    # Flip the label
    label[1] = image.shape[1] - label[1]


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/LANES/100_NL_F_FA/l5_f0123.jpg")

    # PATH_SAVE = Path("../../../../../data/5_model_output/tries/flipped_images")

    # LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    # LABEL = np.array([83, 2903])
    LABEL = np.array([136, 2774])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE)).astype(np.float)

    # Plot the original image
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    cv2.imshow(" Image", IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Flip the image
    flip_image(IMAGE, LABEL)

    # Register the flipped image
    # cv2.imwrite(str(PATH_SAVE / "flipped_l5_f0123.jpg"), IMAGE)

    # Plot the flipped image
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    cv2.imshow("Flipped Image", IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
