import numpy as np
from tensorflow import GradientTape
from tensorflow import norm, convert_to_tensor
from tensorflow.keras.losses import binary_crossentropy


def create_label(labels):
    """
    Transform the labels' batch that are integers to list of zeros where there is a one in the right place.
    """
    nb_labels = len(labels)
    full_labels = np.zeros((len(labels), 10))
    for idx_label in range(nb_labels):
        full_labels[idx_label, int(labels[idx_label])] = 1

    return convert_to_tensor(full_labels, dtype=np.float32)


def get_loss(model, inputs, labels):
    """
    Get the loss and the gradient of every layer.
    """
    full_labels = create_label(labels)
    with GradientTape() as tape:
        loss_value = cross_loss(model, inputs, full_labels)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(model, inputs, full_labels):
    """
    Get the loss of the model.
    """
    outputs = model(inputs)

    return norm(binary_crossentropy(outputs, full_labels)) ** 2


def evaluate(model, inputs, labels):
    """
    Evaluate the model without back propagation.
    """
    full_labels = create_label(labels)

    loss_value = cross_loss(model, inputs, full_labels)

    return loss_value


if __name__ == "__main__":
    LABELS = [2, 3]
    FULL_LABELS = create_label(LABELS)

    print(FULL_LABELS)
