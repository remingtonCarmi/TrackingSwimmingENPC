"""
This module performs data augmentation.
"""
from pathlib import Path
import numpy as np
import cv2
import numpy.random as rd

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale


def augment(image):
    """
    Change the color channel of the image.

    Args:
        image (array): the input image.
    """
    # Change the color
    image *= rd.uniform(0, 1.5, 3)
    image += rd.uniform(-15, 15, 3) + rd.uniform(-25, 25, image.shape)

    np.clip(image, 0, 255, image)


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/100NL_FAF/l8_f1054.jpg")
    # PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/50_Br_M_SF_1/l1_f0339.jpg")
    PATH_IMAGE = Path("../../../../../data/5_model_output/tries/flipped_images/flipped_l5_f0123.jpg")

    # PATH_SAVE = Path("../../../../../data/5_model_output/tries/augmented_images")

    SCALE = 35
    VIDEO_LENGTH = 25
    # VIDEO_LENGTH = 50.97291666666667 - -1
    # VIDEO_LENGTH = 28.3
    VIDEO_LENGTH = 50.98645833333333 - 6.204166666666667
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    # LABEL = np.array([83, 2903])
    LABEL = np.array([136, 2774])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE)).astype(np.float)

    # Rescale the image
    (RESCALED_IMAGE, RESCALED_LABEL) = rescale(IMAGE, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])

    # Plot rescaled image
    RESCALED_IMAGE[RESCALED_LABEL[0], RESCALED_LABEL[1]] = [0, 0, 255]
    cv2.imshow("Rescaled Image", RESCALED_IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for idx_try in range(5):
        COPY_ORIGINAL = IMAGE.copy()
        # Augment the image
        augment(COPY_ORIGINAL)

        # Rescale the augmented image
        (RESCALED_AUGM_IMAGE, RESCALED_AUGM_LABEL) = rescale(COPY_ORIGINAL, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])
        # cv2.imwrite(str(PATH_SAVE / "augmented_l5_f0123_{}.jpg".format(idx_try)), COPY_ORIGINAL)

        # Plot rescaled augmented image
        RESCALED_AUGM_IMAGE[RESCALED_AUGM_LABEL[0], RESCALED_AUGM_LABEL[1]] = [0, 0, 255]
        cv2.imshow("Rescaled and Augmented Image", RESCALED_AUGM_IMAGE.astype("uint8"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
