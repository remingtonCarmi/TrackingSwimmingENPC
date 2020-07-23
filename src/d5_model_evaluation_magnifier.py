"""
This module computes the predictions by evaluating two models : a first rough model and a second more precise model.
"""
# To slice the lane
from src.d5_model_evaluation.slice_lane.slice_lanes import slice_lane

# To merge the predictions
from src.d5_model_evaluation.merge_predictions import merge_predictions


def evaluate_model(model_rough, model_tight, lane, label, window_sizes, recoveries):
    """
    Evaluate the models to produce a prediction on the location of the head.

    Args:
        model_rough (Trained Model): the first rough model.

        model_tight (Trained Model): the second tight model.

        lane (array): a lane.

        label (array): the label linked with the lane.

        window_sizes (array of integer): the two window sizes.

        recoveries (array of integer): the two recoveries.

    Returns:
        index_tight_predictions (array of integer): the list of the columns that might contain a head.

        index_regression_pred (integer): the predicted position of the head.
    """
    # -- Get the first rough predictions -- #
    # Get the large sub-images
    (sub_lanes_rough, sub_labels_rough, lane_iterator_rough) = slice_lane(lane, label, window_sizes[0], recoveries[0])
    # Compute rough predictions
    rough_predictions = model_rough(sub_lanes_rough)

    # -- Merge the rough predictions -- #
    (index_rough_predictions, index_rough_regression_pred) = merge_predictions(rough_predictions, lane_iterator_rough)
    (left_rough_pred, right_rough_pred) = (index_rough_predictions[0], index_rough_predictions[-1] + 1)
    print("Left rough pred", left_rough_pred, "Right rough pred", right_rough_pred)
    print("Prediction rough", index_rough_predictions)

    # -- Get the second tight predictions -- #
    # Adapt the label
    label_tight = label.copy()
    label_tight[1] -= left_rough_pred
    # Get the tight sub-images
    print("Tight")
    (sub_lanes_tight, sub_labels_tight, lane_iterator_tight) = slice_lane(lane[:, left_rough_pred: right_rough_pred], label_tight, window_sizes[1], recoveries[1])
    # Compute tight predictions
    tight_predictions = model_tight(sub_lanes_tight)
    print("Prediction tight value", tight_predictions)

    # -- Merge the rough predictions -- #
    (index_tight_predictions, index_regression_pred) = merge_predictions(tight_predictions, lane_iterator_tight)

    return index_tight_predictions + left_rough_pred, int(index_regression_pred) + left_rough_pred
