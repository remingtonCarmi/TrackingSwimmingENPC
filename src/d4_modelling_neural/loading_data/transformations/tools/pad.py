"""
This module pads the input image with zeros.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import PaddingError

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale


def pad(image, dimensions, label):
    """
    Pad the image with zeros.

    Args:
        image (array): the input image.

        dimensions (list of two integers): [vertic_dimension, horiz_dimension] for padding.

        label (List of 2 integer): the position of the head in pixels. [vertical, horizontal]

    Returns:
        padded_image (array): the padded image.

        label (List of 2 integer): the position of the head in pixels in the modified image. [vertical, horizontal]
    """
    # Compute padding dimensions
    vertic_pad = dimensions[0] - image.shape[0]
    horiz_pad = dimensions[1] - image.shape[1]

    # Check if the padding is possible
    if vertic_pad < 0 or horiz_pad < 0:
        raise PaddingError(image.shape, dimensions)

    # Translate the label
    padded_label = label + np.array([vertic_pad // 2, horiz_pad // 2])
    padded_image = np.pad(image, ((vertic_pad // 2, vertic_pad - vertic_pad // 2), (horiz_pad // 2, horiz_pad - horiz_pad // 2), (0, 0)))

    return padded_image, padded_label


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../../data/4_model_output/tries/transformed_images/padded_l1_f0107.jpg")

    SCALE = 35
    VIDEO_LENGTH = 25
    # VIDEO_LENGTH = 50.97291666666667 - -1
    VIDEO_LENGTH = 28.3
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387])
    # LABEL = np.array([83, 644])
    # LABEL = np.array([53, 1003])
    LABEL = np.array([83, 2903])

    try:
        # Load the image
        IMAGE = cv2.imread(str(PATH_IMAGE))

        # Rescale the image
        (RESCALED_IMAGE, RESCALED_LABEL) = rescale(IMAGE, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])

        # Fill the image with black pixels
        (PADDED_IMAGE, PADDED_LABEL) = pad(RESCALED_IMAGE, DIMENSIONS, RESCALED_LABEL)

        # Plot original image
        IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
        cv2.imshow("Image", IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Plot rescaled image
        RESCALED_IMAGE[RESCALED_LABEL[0], RESCALED_LABEL[1]] = [0, 0, 255]
        cv2.imshow("Rescaled Image", RESCALED_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Plot padded image
        PADDED_IMAGE[PADDED_LABEL[0], PADDED_LABEL[1]] = [0, 0, 255]
        # cv2.imwrite(str(PATH_SAVE), PADDED_IMAGE)

        cv2.imshow("Filled Image", PADDED_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except PaddingError as padding_error:
        print(padding_error.__repr__())
