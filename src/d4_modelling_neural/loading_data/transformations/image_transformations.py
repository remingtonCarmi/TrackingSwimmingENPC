"""
This module transform the image.
It standardizes, rescales and fills with black the image with its label.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.exceptions.exception_classes import PaddingError


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


def rescale(image, scale, video_length, label, pixel_vertic_dim):
    """
    Rescale the image to the given scale for the horizontal axis and
    keep the size of the vertical axis.

    Args:
        image (array): the image to rescale.

        scale (integer): the scale in pixels per meter.

        video_length (float): the length in meters of the video.

        label (List of 2 integer): the position of the head in pixels. [vertical, horizontal]

        pixel_vertic_dim (integer): the dimension of the vertical axis that will be returned.

    Returns:
        (array): the rescaled image.

        label (List of 2 integer): the position of the head in pixels in the modified image. [vertical, horizontal]
    """
    # Get the original dimension in pixels
    (original_vertic_dim, original_horiz_dim) = image.shape[: 2]

    # Get the final dimension in pixels of the horizontal axis
    pixels_horiz_dim = int(scale * video_length)
    scale_factor_horiz = pixels_horiz_dim / original_horiz_dim

    # Transform the label on the horizontal axis
    rescaled_label = label.copy()
    rescaled_label[1] = np.floor(label[1] * scale_factor_horiz)

    # Transform the label on the horizontal axis
    scale_factor_vertic = pixel_vertic_dim / original_vertic_dim
    rescaled_label[0] = np.floor(label[0] * scale_factor_vertic)

    # Resize the image
    return cv2.resize(image, (pixels_horiz_dim, pixel_vertic_dim)), rescaled_label.astype(int)


def fill_with_black(image, dimensions, label):
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


def transform_image(image_path, label, scale, video_length, dimensions, standardization=True):
    """
    Transform the image by standardizing, rescaling, padding the image and its label.

    Args:
        image_path (WindowsPath): path that leads to the image.

        label (List of 2 integer): the position of the head in pixels. [vertical, horizontal]

        scale (integer): the number of pixel per meters.

        video_length (float): the length of the video in meters.

        dimensions (list of 2 integers): the final dimensions of the image. [vertical, horizontal]

        standardization (boolean): standardize the lane_magnifier if standardization = True.
            Default value = True

    Returns:
        (array): the image that have been standardized, rescaled and padded.

        (list of 2 integers): the position of the head in pixels. [vertical, horizontal]
    """
    # Get the image, standardize, rescale and pad it
    image = cv2.imread(str(image_path))
    if standardization:
        image = standardize(image)
    (image, rescaled_label) = rescale(image, scale, video_length, label, dimensions[0])
    return fill_with_black(image, dimensions, rescaled_label)


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../data/4_model_output/tries/scaled_images/scaled_l1_f0107.jpg")

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
        (PADDED_IMAGE, PADDED_LABEL) = fill_with_black(RESCALED_IMAGE, DIMENSIONS, RESCALED_LABEL)

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
        cv2.imshow("Filled Image", PADDED_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # All in one function
        (FINAL_IMAGE, FINAL_LABEL) = transform_image(PATH_IMAGE, LABEL, SCALE, VIDEO_LENGTH, DIMENSIONS, standardization=False)
        # Save the transformed image
        # cv2.imwrite(str(PATH_SAVE), FINAL_IMAGE)

        # Plot the final image
        FINAL_IMAGE[FINAL_LABEL[0], FINAL_LABEL[1]] = [0, 0, 255]
        print(FINAL_LABEL)
        cv2.imshow("Final Image", FINAL_IMAGE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except PaddingError as padding_error:
        print(padding_error.__repr__())
