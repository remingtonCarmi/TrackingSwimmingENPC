"""
This module computes the prediction of the location of the head of the swimmer given a lane.
"""
from pathlib import Path
import numpy as np
from scipy.special import softmax


def merge_predictions(lane_magnifier, predictions):
    """
    Merge the predictions in order to have the biggest intersection of the biggest union.

    Args:
        lane_magnifier (ImageMagnifier): the lane.

        predictions (array of 2 dimensions, list of [float, float, float]):
            the list of the PREDICTIONS of all the sub-images : [is_in_sub_image, is_not_in_sub_image, column].

    Returns:
        (Interval): the interval that corresponds to the prediction.
    """
    indicative_function = 1 - np.argmax(predictions[:, : 2], axis=1)
    indexes_head = np.where(indicative_function == 1)[0]

    prediction_manager = PredictionManager()

    # For all the indexes where the head has been predicted, the color is change.
    for idx_sub_image in indexes_head:
        (begin_limit, end_limit) = lane_magnifier.get_limits(idx_sub_image)
        prediction_manager.add_prediction(begin_limit, end_limit)

    return prediction_manager.get_bigest_interval()


if __name__ == "__main__":
    # -- Imports -- #
    import cv2

    # To get the image
    from src.d4_modelling_neural.loading_data.transformations.image_transformations import transform_image

    # To modify the image
    from src.d4_modelling_neural.magnifier.slice_sample_lane.image_objects.image_magnifier import ImageMagnifier

    # -- Get the ImageMagnifier i.e. the lane_magnifier -- #
    PATH_IMAGE = Path("../../data/2_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    (LANE, LABEL) = transform_image(PATH_IMAGE, np.array([43, 387]), 35, 25, [108, 1820], False, False, True)
    IMAGE_MAGNIFIER = ImageMagnifier(LANE, LABEL, 200, 10)

    # -- Set the PREDICTIONS -- #
    PREDS = [[0, 1, -1]] * (len(IMAGE_MAGNIFIER) // 2 - 1)
    PREDS.append([1, 0, 100])
    PREDS.append([100, 9, 170])
    PREDS.extend([[0, 1, -1]] * (len(IMAGE_MAGNIFIER) - 1 - len(IMAGE_MAGNIFIER) // 2))
    print("Predictions", PREDS)
    print("len(PREDS)", len(PREDS))

    # -- Plot the lane_magnifier with the prediction -- #
    pred_interval = merge_predictions(IMAGE_MAGNIFIER, np.array(PREDS))
    print(pred_interval)
