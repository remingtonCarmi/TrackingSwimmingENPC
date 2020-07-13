"""
This module transform the image.
It standardizes, rescales and fills with black the image with its label.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import PaddingError

# To standardize the image
from src.d4_modelling_neural.loading_data.transformations.tools.standardize import standardize

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale

# To pad the image
from src.d4_modelling_neural.loading_data.transformations.tools.pad import pad


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
    return pad(image, dimensions, rescaled_label)


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid0/l1_f0275.jpg")
    # PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/vid1/l1_f0107.jpg")
    # PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/LANES/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../data/4_model_output/tries/transformed_images/transformed_l1_f0107.jpg")

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
