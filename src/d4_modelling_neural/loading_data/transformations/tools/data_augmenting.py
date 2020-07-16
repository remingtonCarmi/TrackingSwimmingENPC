"""
This module performs data augmentation.
"""
from pathlib import Path
import numpy as np
import cv2
import numpy.random as rd

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale


def change_one_color(image, color, factor):
    """
    Change the color of one color channel choosing to add or to withdraw the factor
    at random. This is done inplace

    Args:
        image (array): the input image.

        color (integer): the index of the color.

        factor (integer): the amount of intensity to withdraw or to add.
    """
    pos_neg = rd.randint(0, 2)

    if pos_neg:
        # Add factor to every pixel of the color
        indexes = np.where(255 - factor >= image[:, :, color])
        image[:, :, color][indexes] += factor

        # Put again the black padding in black
        indexes = np.where(image[:, :, color] == factor)
        image[:, :, color][indexes] = 0

    else:
        # Withdraw factor to every pixel of the color
        indexes = np.where(image[:, :, color] >= factor)
        image[:, :, color][indexes] -= factor


def augment(image):
    """
    Change the color channel of the image.

    Args:
        image (array): the input image.

    Returns:
        augmented_image (array): the augmented image.
    """
    # Copy the original image and the original label
    augmented_image = image.copy()

    # Change one color
    nb_colors = rd.randint(1, 4)
    colors = rd.choice(range(3), nb_colors, replace=False)
    factors = [rd.randint(0, 60), rd.randint(0, 40), rd.randint(0, 60)]

    for color in colors:
        change_one_color(augmented_image, color, factors[color])

    return augmented_image


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../../data/4_model_output/tries/augmented_images")

    SCALE = 35
    VIDEO_LENGTH = 25
    # VIDEO_LENGTH = 50.97291666666667 - -1
    # VIDEO_LENGTH = 28.3
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    # LABEL = np.array([83, 2903])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE))

    # Rescale the image
    (RESCALED_IMAGE, RESCALED_LABEL) = rescale(IMAGE, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])

    # Plot rescaled image
    RESCALED_IMAGE[RESCALED_LABEL[0], RESCALED_LABEL[1]] = [0, 0, 255]
    cv2.imshow("Rescaled Image", RESCALED_IMAGE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for idx_try in range(5):
        # Augment the image
        (AUGMENTED_IMAGE) = augment(IMAGE)

        # Rescale the augmented image
        (RESCALED_AUGM_IMAGE, RESCALED_AUGM_LABEL) = rescale(AUGMENTED_IMAGE, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])
        # cv2.imwrite(str(PATH_SAVE / "augmented_l1_f0275_{}.jpg".format(idx_try)), RESCALED_AUGM_IMAGE)

        # Plot rescaled augmented image
        RESCALED_AUGM_IMAGE[RESCALED_AUGM_LABEL[0], RESCALED_AUGM_LABEL[1]] = [0, 0, 255]
        cv2.imshow("Rescaled and Augmented Image", RESCALED_AUGM_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
