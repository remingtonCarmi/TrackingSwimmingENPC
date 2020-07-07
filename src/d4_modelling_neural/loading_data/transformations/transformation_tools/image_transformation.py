"""
This module transform the images.
It standardizes, rescales and fills with black the images.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.transformation_tools.exceptions.exception_classes import PaddingError


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


def rescale(image, scale, length_video):
    """
    Rescale the image to the given scale.

    Args:
        image (array): the image to rescale.

        scale (integer): the scale in pixels per meter.

        length_video (float): the length in meters of the video.

    Returns:
        (array): the rescaled image.
    """
    # Get the original dimension in pixels
    (original_vertic_dim, original_horiz_dim) = image.shape[: 2]

    # Get the final dimension in pixels
    pixels_horiz_dim = int(scale * length_video)
    pixels_vertic_dim = int(original_vertic_dim * pixels_horiz_dim / original_horiz_dim)

    # Resize the image
    return cv2.resize(image, (pixels_horiz_dim, pixels_vertic_dim))


def fill_with_black(image, dimensions):
    """
    Pad the image with zeros.

    Args:
        image (array): the input image.

        dimensions (list of two integers): [vertic_dimension, horiz_dimension] for padding.

    Returns:
        (array): the padded image.
    """
    vertic_pad = dimensions[0] - image.shape[0]
    horiz_pad = dimensions[1] - image.shape[1]
    print("half padding", horiz_pad // 2)
    if vertic_pad < 0 or horiz_pad < 0:
        raise PaddingError(image.shape, dimensions)
    return np.pad(image, ((vertic_pad // 2, vertic_pad - vertic_pad // 2), (horiz_pad // 2, horiz_pad - horiz_pad // 2), (0, 0)))


if __name__ == "__main__":
    # Path to the image
    # PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/lanes/tries/100NL_FAF/l1_f0100.jpg")
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/lanes/tries/vid0/l1_f0275.jpg")

    try:
        # Load the image
        IMAGE = cv2.imread(str(PATH_IMAGE))

        # Rescale the image
        # RESCALED_IMAGE = rescale(IMAGE, 35, 50.97291666666667 - -1)
        RESCALED_IMAGE = rescale(IMAGE, 35, 25)

        # Fill the image with black pixels
        DIMENSIONS = [110, 1820]
        FILLED_IMAGE = fill_with_black(RESCALED_IMAGE, DIMENSIONS)

        cv2.imshow("Image", IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Rescaled Image", RESCALED_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Filled Image", FILLED_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(np.sum(FILLED_IMAGE[:, -1]))
    except PaddingError as padding_error:
        print(padding_error.__repr__())
