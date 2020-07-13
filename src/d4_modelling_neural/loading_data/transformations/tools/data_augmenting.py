"""
This module performs data augment.
"""
from pathlib import Path
import numpy as np
import cv2

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale


def augment(image, label, random_seed):
    return image, label


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../../data/4_model_output/tries/transformed_images/augmented_l1_f0107.jpg")

    SCALE = 35
    VIDEO_LENGTH = 25
    # VIDEO_LENGTH = 50.97291666666667 - -1
    VIDEO_LENGTH = 28.3
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    LABEL = np.array([83, 2903])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE))

    # Augment the image
    (AUGMENTED_IMAGE, AUGMENTED_LABEL) = augment(IMAGE, LABEL, 0)

    # Rescale the image
    (RESCALED_IMAGE, RESCALED_LABEL) = rescale(AUGMENTED_IMAGE, SCALE, VIDEO_LENGTH, AUGMENTED_LABEL, DIMENSIONS[0])

    # Plot original image
    cv2.imshow("Image", IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plot the augmented image
    AUGMENTED_IMAGE[AUGMENTED_LABEL[0], AUGMENTED_LABEL[1]] = [0, 0, 255]
    cv2.imshow("Augmented Image", AUGMENTED_IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plot rescaled image
    RESCALED_IMAGE[RESCALED_LABEL[0], RESCALED_LABEL[1]] = [0, 0, 255]
    cv2.imshow("Rescaled Image", RESCALED_IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


