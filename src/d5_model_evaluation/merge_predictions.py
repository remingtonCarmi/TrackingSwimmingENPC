"""
This module computes the prediction of the location of the head of the swimmer given a lane.
"""
import numpy as np


def merge_predictions(predictions, lane_iterator):
    """
    Merge the predictions in order to have the biggest intersection of the biggest union.

    Args:
        predictions (array of 2 dimensions, list of [float, float, float]):
            the list of the predictions of all the sub-images : [is_in_sub_image, is_not_in_sub_image, column].

        lane_iterator (LaneIterator): an object that gives the limits of the sliced sub-images.

    Returns:
        (array): the list of the index where the head can be.
    """
    # Get the index where the head can be
    indicative_function = 1 - np.argmax(predictions[:, : 2], axis=1)
    indexes_head = np.where(indicative_function == 1)[0]

    # Will count the number of time that a head can be in a specific column
    witness_prediction = np.zeros(lane_iterator.image_horiz_size)

    for idx_sub_image in indexes_head:
        (begin_limit, end_limit) = lane_iterator.get_limits(idx_sub_image)
        witness_prediction[begin_limit: end_limit] += 1

    # Return the columns with the highest probability to have a head
    return np.where(witness_prediction == max(witness_prediction))[0]


if __name__ == "__main__":
    # To have a lane iterator
    from src.d5_model_evaluation.slice_lane.image_magnifier.lane_iterator import LaneIterator

    NB_SUB_IMAGES = 10
    WINDOW_SIZE = 200
    RECOVERY = 100
    IMAGE_HORIZ_SIZE = 1000

    # Set the lane iterator
    LANE_ITERATOR = LaneIterator(NB_SUB_IMAGES, WINDOW_SIZE, RECOVERY, IMAGE_HORIZ_SIZE)

    # -- Set the PREDICTIONS -- #
    PREDS = [[0, 1, -1]] * (NB_SUB_IMAGES // 2 - 1)
    PREDS.append([1, 0, 100])
    PREDS.append([100, 9, 170])
    PREDS.extend([[0, 1, -1]] * (NB_SUB_IMAGES - 1 - NB_SUB_IMAGES // 2))
    print("Predictions", PREDS)
    print("len(PREDS)", len(PREDS))

    # -- Plot the lane_magnifier with the prediction -- #
    pred_interval = merge_predictions(np.array(PREDS), LANE_ITERATOR)
    print(pred_interval)
