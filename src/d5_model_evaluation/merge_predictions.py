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

        (integer): the predicted position of the head.
    """
    # Get the index where the head can be
    indicative_function = 1 - np.argmax(predictions[:, : 2], axis=1)
    indexes_head = np.where(indicative_function == 1)[0]

    # Will count the number of time that a head can be in a specific column
    witness_classification = np.zeros(lane_iterator.image_horiz_size)
    witness_regression = np.zeros(lane_iterator.image_horiz_size)

    for idx_sub_image in indexes_head:
        (begin_limit, end_limit) = lane_iterator.get_limits(idx_sub_image)
        witness_classification[begin_limit: end_limit] += 1
        witness_regression[np.clip(begin_limit + int(predictions[idx_sub_image, -1]), 0, lane_iterator.image_horiz_size)] += 1

    # Take the columns with the highest probabilities
    classification_columns = np.where(witness_classification == max(witness_classification))[0]

    # Take the column with the highest probability
    regression_columns = np.where(witness_regression > 0)[0]

    # If there is any column return the middle of the initial image
    if len(regression_columns) == 0:
        final_regression = lane_iterator.image_horiz_size // 2
    else:
        final_regression = np.max(regression_columns)
    # Return the columns with the highest probability to have a head
    return classification_columns, final_regression


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
    (pred_interval, pred_head) = merge_predictions(np.array(PREDS), LANE_ITERATOR)
    print(pred_interval)
    print(pred_head)
