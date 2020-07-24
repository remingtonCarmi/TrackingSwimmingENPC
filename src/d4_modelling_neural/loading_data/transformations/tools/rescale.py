"""
This module rescales the input image.
"""
from pathlib import Path
import cv2
import numpy as np


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
    rescaled_label[1] = int(np.floor(label[1] * scale_factor_horiz))

    # Transform the label on the horizontal axis
    scale_factor_vertic = pixel_vertic_dim / original_vertic_dim
    rescaled_label[0] = int(np.floor(label[0] * scale_factor_vertic))

    # Resize the image
    return cv2.resize(image, (pixels_horiz_dim, pixel_vertic_dim)), rescaled_label


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid0/l1_f0275.jpg")
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0107.jpg")
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/100NL_FAF/l8_f1054.jpg")
    PATH_IMAGE = Path("../../../../../data/2_intermediate_top_down_lanes/lanes/tries/50_Br_M_SF_1/l1_f0339.jpg")
    PATH_IMAGE = Path("../../../../../data/5_model_output/tries/augmented_images/augmented_l5_f0123_2.jpg")

    # PATH_SAVE = Path("../../../../../data/5_model_output/tries/rescaled_images/rescaled_l5_f0123.jpg")

    SCALE = 35
    VIDEO_LENGTH = 25
    VIDEO_LENGTH = 50.97291666666667 - -1
    VIDEO_LENGTH = 28.3
    VIDEO_LENGTH = 50.98645833333333 - 6.204166666666667
    DIMENSIONS = [108, 1820]
    LABEL = np.array([43, 387])
    LABEL = np.array([83, 644])
    LABEL = np.array([53, 1003])
    LABEL = np.array([83, 2903])
    LABEL = np.array([136, 2774])

    # Load the image
    IMAGE = cv2.imread(str(PATH_IMAGE)).astype(np.float)

    # Rescale the image
    (RESCALED_IMAGE, RESCALED_LABEL) = rescale(IMAGE, SCALE, VIDEO_LENGTH, LABEL, DIMENSIONS[0])

    # Plot original image
    IMAGE[LABEL[0], LABEL[1]] = [0, 0, 255]
    cv2.imshow("Image", IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plot rescaled image
    RESCALED_IMAGE[RESCALED_LABEL[0], RESCALED_LABEL[1]] = [0, 0, 255]
    # cv2.imwrite(str(PATH_SAVE), RESCALED_IMAGE)
    cv2.imshow("Rescaled Image", RESCALED_IMAGE.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
