from pathlib import Path
import numpy as np
from scipy.special import softmax


def add_prediction(lane_magnifier, predictions):
    """
    Add the prediction to the lane.
    That is to say, darker the sub-images that where predicted with a head
    and add a red column at the predicted position of the head.

    Args:
        lane_magnifier (ImageMagnifier): the lane.

        predictions (array of 2 dimensions, list of [float, float, float]):
            the list of the PREDICTIONS of all the sub-images : [is_in_sub_image, is_not_in_sub_image, column].

    Returns:
        LANE_PRED (array): the lane with the added PREDICTIONS.
    """
    # Get the lane_magnifier
    lane_pred = lane_magnifier.lane

    # Find the indexes where the head has been predicted
    indicative_function = 1 - np.argmax(predictions[:, : 2], axis=1)
    indexes_head = np.where(indicative_function == 1)[0]

    print("Predictions", predictions)
    print("Head predictions indexes", indexes_head)
    # print("Head predictions values", np.array(predictions)[indexes_head, :2])

    print("Number of head predicted : ", len(indexes_head))

    # For all the indexes where the head has been predicted, the color is change.
    for idx_sub_image in indexes_head:
        (begin_limit, end_limit) = lane_magnifier.get_limits(idx_sub_image)

        # Darker the pixels
        trust = (2 * softmax(predictions[idx_sub_image, :2])[0] - 1) ** 2
        factor = int(np.round(50 * trust))
        print("Factor", factor)
        # On the second channel
        high_indexes_1 = np.where(lane_pred[:, begin_limit: end_limit, 1] > factor)
        lane_pred[:, begin_limit: end_limit, 1][high_indexes_1] -= factor

        # --- Visualize the column --- #
        lane_pred[:, begin_limit + int(np.floor(predictions[idx_sub_image, 2]))] = [0, 0, 255]

    return lane_pred.astype('uint8')


if __name__ == "__main__":
    # -- Imports -- #
    import cv2

    # To get the image
    from src.d4_modelling_neural.loading_data.transformations.image_transformations import transform_image

    # To modify the image
    from src.d4_modelling_neural.magnifier.slice_sample_lane.image_objects.image_magnifier import ImageMagnifier

    # -- Get the ImageMagnifier i.e. the lane_magnifier -- #
    PATH_IMAGE = Path("../../data/2_intermediate_top_down_lanes/LANES/tries/100NL_FAF/l8_f1054.jpg")
    (LANE, LABEL) = transform_image(PATH_IMAGE, np.array([43, 387]), 35, 25, [110, 1820], augmentation=False, standardization=False)
    IMAGE_MAGNIFIER = ImageMagnifier(LANE, LABEL, 200, 10)

    # -- Set the PREDICTIONS -- #
    PREDS = [[0, 1, -1]] * (len(IMAGE_MAGNIFIER) // 2 - 1)
    PREDS.append([1, 0, 100])
    PREDS.append([100, 9, 170])
    PREDS.extend([[0, 1, -1]] * (len(IMAGE_MAGNIFIER) - 1 - len(IMAGE_MAGNIFIER) // 2))
    print("Predictions", PREDS)
    print("len(PREDS)", len(PREDS))

    # -- Plot the lane_magnifier with the prediction -- #
    LANE_PRED = add_prediction(IMAGE_MAGNIFIER, np.array(PREDS))
    cv2.imshow("Image with PREDICTIONS", LANE_PRED)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
