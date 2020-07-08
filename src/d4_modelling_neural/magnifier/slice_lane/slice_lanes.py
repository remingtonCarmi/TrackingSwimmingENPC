"""
This module slices a list of lanes with its labels.
"""
from pathlib import Path
import cv2
import numpy as np
from src.d4_modelling_neural.magnifier.slice_lane.image_magnifier.image_magnifier import ImageMagnifier


def slice_lanes(lanes, labels, window_size, recovery):
    """
    Slices the list of lanes with its labels.
    The lanes are sliced one after the order.

    Args:
        lanes (array of 4 dimensions): the list of lanes.

        labels (array of 3 dimensions): the list of labels.

        window_size (integer): the width of the sub-images.

        recovery (integer): the number of pixels to be taken twice per sub-image.

    Returns:
        sub_lanes (array of 4 dimensions): the list of sub-image.

        (array: (integer, integer, integer)): (is_in_image, is_not_in_image, column)
            column is the index of the column of pixel where the head is located.
            If present is False, column = -1
    """
    nb_lanes = len(lanes)
    sub_lanes = []
    sub_labels = []
    for idx_lanes in range(nb_lanes):
        magnifier = ImageMagnifier(lanes[idx_lanes], labels[idx_lanes], window_size, recovery)

        for (sub_lane, sub_label) in magnifier:
            sub_lanes.append(sub_lane.astype(dtype='float32'))
            sub_labels.append(sub_label)

    return np.array(sub_lanes), np.array(sub_labels).astype(dtype='float32')


if __name__ == "__main__":
    # Data
    PATH_IMAGE1 = Path("../../../../data/1_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0297.jpg")
    PATH_IMAGE2 = Path("../../../../data/4_model_output/tries/scaled_images/scaled_l1_f0275.jpg")

    LANES = np.array([cv2.imread(str(PATH_IMAGE1)), cv2.imread(str(PATH_IMAGE2))])
    LABELS = np.array([[54, 1560], [49, 648]])
    WINDOW_SIZE = 200
    RECOVERY = 100

    # Slice the image
    (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)

    # Plot the sub-lanes
    NB_SUB_LANES = len(SUB_LANES)
    for idx_image in range(NB_SUB_LANES):
        print("Image n° {}. Present = {}".format(idx_image, SUB_LABELS[idx_image][0]))
        if SUB_LABELS[idx_image][0]:
            SUB_LANES[idx_image][:, int(SUB_LABELS[idx_image][2])] = [0, 0, 255]
        cv2.imshow("Image n' {}. Present = {}".format(idx_image, SUB_LABELS[idx_image][0]), SUB_LANES[idx_image].astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
