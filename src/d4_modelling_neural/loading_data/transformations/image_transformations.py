"""
This module transform the image.
It standardizes, rescales and fills with black the image with its label.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import PaddingError

# To flip the image
from src.d4_modelling_neural.loading_data.transformations.tools.flip import flip_image

# To augment the image
from src.d4_modelling_neural.loading_data.transformations.tools.data_augmenting import augment

# To standardize the image
from src.d4_modelling_neural.loading_data.transformations.tools.standardize import standardize

# To rescale the image
from src.d4_modelling_neural.loading_data.transformations.tools.rescale import rescale

# To pad the image
from src.d4_modelling_neural.loading_data.transformations.tools.pad import pad


def transform_image(image_path, label, scale, video_length, dimensions, standardization, augmentation, flip):
    """
    Transform the image by augmenting, standardizing, rescaling, padding the image and its label.

    Args:
        image_path (WindowsPath): path that leads to the image.

        label (List of 3 float): the position of the head in pixels. [vertical, horizontal, swimming_way]

        scale (integer): the number of pixel per meters.

        video_length (float): the length of the video in meters.

        dimensions (list of 2 integers): the final dimensions of the image. [vertical, horizontal]

        standardization (boolean): standardize the lane, if standardization = True.

        augmentation (boolean): augment the lane, if augmentation = True.

        flip (boolean): flip the image if the swimmer goes to the left, if flip = True.

    Returns:
        (array): the image that have been standardized, rescaled and padded.

        (list of 2 integers): the position of the head in pixels. [vertical, horizontal]
    """
    # Load the image and define the position label
    image = cv2.imread(str(image_path)).astype(np.float)
    pos_label = label[:-1]

    # Flip the image if flip = True and if it has to be flipped.
    if flip and label[-1] == -1:
        flip_image(image, pos_label)

    # Augment the image
    if augmentation:
        augment(image)

    # Standardize the image
    if standardization:
        standardize(image)

    # Rescale the image
    (image, rescaled_label) = rescale(image, scale, video_length, pos_label, dimensions[0])

    # Pad the image
    return pad(image, dimensions, rescaled_label)


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid0/l1_f0275.jpg")
    PATH_IMAGE = Path("../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0107.jpg")
    PATH_IMAGE = Path("../../../../data/2_intermediate_top_down_lanes/lanes/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../data/2_intermediate_top_down_lanes/lanes/tries/50_Br_M_SF_1/l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../data/5_model_output/tries/transformed_images/transformed_l1_f0275.jpg")

    SCALE = 35
    VIDEO_LENGTH = 25
    VIDEO_LENGTH = 50.97291666666667 - -1
    VIDEO_LENGTH = 28.3
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387, 1])
    LABEL = np.array([83, 644, 1])
    LABEL = np.array([53, 1003, 1])
    LABEL = np.array([83, 2903, -1])

    try:
        # All in one function
        (FINAL_IMAGE, FINAL_LABEL) = transform_image(PATH_IMAGE, LABEL, SCALE, VIDEO_LENGTH, DIMENSIONS, False, True, True)
        # Save the transformed image
        # cv2.imwrite(str(PATH_SAVE), FINAL_IMAGE)

        # Plot the final image
        FINAL_IMAGE[FINAL_LABEL[0], FINAL_LABEL[1]] = [0, 0, 255]
        print(FINAL_LABEL)
        cv2.imshow("Final Image", FINAL_IMAGE.astype("uint8"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except PaddingError as padding_error:
        print(padding_error.__repr__())
