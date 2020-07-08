"""
This module computes the loss of the model.
"""
import numpy as np

# To compute the gradient
from tensorflow import GradientTape

# To compute the gradient while computing a sum
from tensorflow import reduce_sum

# For the classification problem
from tensorflow.keras.losses import binary_crossentropy


def get_loss(model, sub_lanes, sub_labels, trade_off):
    """
    Get the loss and the gradient of every layer.

    Args:
        model (TensorFlow Model): the model.

        sub_lanes (array of 4 dimensions): the list of sub-lanes.

        sub_labels (array of 4 dimensions, list of : (integer, integer, integer)):
            (is_in_image, is_not_in_image, column), column is the index of the column of pixel is the head is located.
            If present is False, column = -1

        trade_off (float): the trade off between the classification loss and the regression loss.

    Returns:
        (TensorFlow object): the gradient of each layer of the model.

        loss_value (float): the value of the loss.

        predictions (tensor of 2 dimensions): list of [pred_is_in_image, pred_is_not_in_image, pred_column]
    """
    with GradientTape() as tape:
        # Evaluate with the model
        predictions = model(sub_lanes)

        # Compute the loss value
        loss_value = compute_loss(sub_labels, predictions, trade_off)

    return tape.gradient(loss_value, model.trainable_variables), loss_value, predictions


def compute_loss(sub_labels, predictions, trade_off):
    """
    Compute the loss.

    Args:
        sub_labels (array of 4 dimensions, list of : (integer, integer, integer)):
            (is_in_image, is_not_in_image, column), column is the index of the column of pixel is the head is located.
            If present is False, column = -1

        predictions (tensor of 2 dimensions): list of [pred_is_in_image, pred_is_not_in_image, pred_column]

        trade_off (float): the trade off between the classification loss and the regression loss.

    Returns:
        loss_value (float): the value of the loss.
    """
    # Loss for the classification problem
    cross_entropy_loss = classification_loss(sub_labels[:, :2], predictions[:, :2])
    # Loss for the regressions problem
    l2_loss = regression_loss(sub_labels[:, 2], predictions[:, 2])

    # Sum the losses
    return cross_entropy_loss + trade_off * l2_loss


def classification_loss(labels, predictions):
    """
    Compute the binary cross entropy loss with logits.

    Args:
        labels (array, 2 dimension): list of list of 2 elements: [is_in_sub_image, is_not_in_sub_image].

        predictions (array, 2 dimensions): list of list of 2 elements: [is_in_sub_image, is_not_in_sub_image].

    Returns:
        (float): the cross entropy loss.
    """
    return reduce_sum(binary_crossentropy(labels, predictions, from_logits=True))


def regression_loss(labels, predictions):
    """
    Compute the mean square error loss.

    Args:
        labels (array, 1 dimension): the column number.

        predictions (array, 1 dimension): the predicted column number.

    Returns:
        (float): the MSE.
    """
    nb_sub_images = len(predictions)
    # Sum of the errors
    sum_reg = 0

    for idx_sub_image in range(nb_sub_images):
        if labels[idx_sub_image] >= 0:
            # L2 loss update
            sum_reg += (predictions[idx_sub_image] - labels[idx_sub_image]) ** 2

    return sum_reg


def evaluate_loss(model, sub_lanes, sub_labels, trade_off):
    """
    Get the loss and the gradient of every layer.

    Args:
        model (TensorFlow Model): the model.

        sub_lanes (array of 4 dimensions): the list of sub-lanes.

        sub_labels (array of 4 dimensions, list of : (integer, integer, integer)):
            (is_in_image, is_not_in_image, column), column is the index of the column of pixel is the head is located.
            If present is False, column = -1

        trade_off (float): the trade off between the classification loss and the regression loss.

    Returns:
        (float): the value of the loss.

        predictions (tensor of 2 dimensions): list of [pred_is_in_image, pred_is_not_in_image, pred_column]
    """
    # Evaluate with the model
    predictions = model(sub_lanes)

    # Compute the loss value
    return compute_loss(sub_labels, predictions, trade_off), predictions


if __name__ == "__main__":
    LABELS = np.array([[0, 1, -1], [0, 1, -1], [1, 0, 10]], dtype=np.float)
    PRED_GOOD = np.array([[-1, 201, 933], [10, 29, -88], [110, -7, 15]], dtype=np.float)
    PRED_BAD = np.array([[1, -201, 933], [100, 2, -193], [1, -77, 190]], dtype=np.float)

    CLASS_LOSS_GOOD = classification_loss(LABELS[:, :2], PRED_GOOD[:, :2])
    CLASS_LOSS_BAD = classification_loss(LABELS[:, :2], PRED_BAD[:, :2])

    REG_LOSS_GOOD = regression_loss(LABELS[:, 2], PRED_GOOD[:, 2])
    REG_LOSS_BAD = regression_loss(LABELS[:, 2], PRED_BAD[:, 2])

    print("Good classification", CLASS_LOSS_GOOD)
    print("Bad classification", CLASS_LOSS_BAD)
    print("Good regression", REG_LOSS_GOOD)
    print("Bad regression", REG_LOSS_BAD)


