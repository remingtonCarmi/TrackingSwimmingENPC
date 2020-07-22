"""
This module slices a list of LANES with its LABELS.
"""
from pathlib import Path
import cv2
import numpy as np
from src.d5_model_evaluation.slice_lane.image_magnifier.image_magnifier import ImageMagnifier
from src.d5_model_evaluation.slice_lane.image_magnifier.lane_iterator import LaneIterator


def slice_lane(lane, label, window_size, recovery):
    """
    Slices the list of lanes with its labels.
    The LANES are sliced one after the order.

    Args:
        lane (array of 3 dimensions): the lane.

        label (array of 2 dimensions): the label.

        window_size (integer): the width of the sub-images.

        recovery (integer): the number of pixels to be taken twice per sub-image.

    Returns:
        sub_lanes (array of 4 dimensions): the list of sub-image.

        (array: (integer, integer, integer)): (is_in_image, is_not_in_image, column)
            column is the index of the column of pixel where the head is located.
            If present is False, column = -1

        (LaneIterator): the lane iterator that enables to have the limits of the sub-images.
    """
    sub_lanes = []
    sub_labels = []

    magnifier = ImageMagnifier(lane, label, window_size, recovery)

    for (sub_lane, sub_label) in magnifier:
        sub_lanes.append(sub_lane.astype(dtype='float32'))
        sub_labels.append(sub_label)

    return np.array(sub_lanes), np.array(sub_labels).astype(dtype='float32'), LaneIterator(len(magnifier), window_size, recovery, lane.shape[1])


if __name__ == "__main__":
    # Data
    PATH_IMAGE = Path("../../../data/5_model_output/tries/transformed_images/transformed_l1_f0275.jpg")
    PATH_IMAGE = Path("../../../data/5_model_output/tries/transformed_images/transformed_l1_f0107.jpg")
    # PATH_IMAGE3 = Path("../../../data/5_model_output/tries/transformed_images/transformed_l8_f1054.jpg")
    # PATH_IMAGE4 = Path("../../../data/5_model_output/tries/transformed_images/transformed_l1_f0339.jpg")

    # PATH_SAVE = Path("../../../../data/5_model_output/tries/sliced_images")

    LANE = cv2.imread(str(PATH_IMAGE))
    LABEL = np.array([49, 648])
    LABEL = np.array([49, 768])
    # LABEL = np.array([53, 950])
    # LABEL = np.array([41, 1163])
    WINDOW_SIZE = 150
    RECOVERY = 75

    # Slice the image
    (SUB_LANES, SUB_LABELS, LANE_ITERATOR) = slice_lane(LANE, LABEL, WINDOW_SIZE, RECOVERY)

    # Plot the sub-LANES
    NB_SUB_LANES = len(SUB_LANES)
    for idx_image in range(NB_SUB_LANES):
        # Save the image
        # PATH_SAVE_SUB_IMAGE = PATH_SAVE / "sliced_l1_f0107_{}.jpg".format(idx_image)
        # cv2.imwrite(str(PATH_SAVE_SUB_IMAGE), SUB_LANES[idx_image])
        print(SUB_LABELS[idx_image])
        print("Image n° {}. Present = {}".format(idx_image, SUB_LABELS[idx_image][0]))
        if SUB_LABELS[idx_image][0]:
            SUB_LANES[idx_image][:, int(SUB_LABELS[idx_image][2])] = [0, 0, 255]
        cv2.imshow("Image n' {}. Present = {}".format(idx_image, SUB_LABELS[idx_image][0]), SUB_LANES[idx_image].astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
