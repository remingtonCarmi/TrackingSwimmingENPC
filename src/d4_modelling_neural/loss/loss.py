import numpy as np
from tensorflow import GradientTape
from tensorflow import norm, convert_to_tensor, reduce_sum
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.losses import mean_squared_error


def create_label(labels, nb_classes):
    """
    Transform the label' batch that are integers to list of zeros where there is a one in the right place.
    """
    nb_labels = len(labels)
    full_labels = np.zeros((len(labels), nb_classes))
    for idx_label in range(nb_labels):
        full_labels[idx_label, int(labels[idx_label])] = 1

    return convert_to_tensor(full_labels, dtype=np.float32)


def get_loss(model, inputs, labels):
    """
    Get the loss and the gradient of every layer.
    """
    with GradientTape() as tape:
        loss_value = cross_loss(model, inputs, labels)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(model, inputs, labels):
    """
    Get the loss of the model.
    """
    outputs = model(inputs)
    return reduce_sum(sparse_categorical_crossentropy(labels, outputs, from_logits=True))


def evaluate_loss(model, data_loader):
    """
    Evaluate the model without back propagation.
    """
    model.trainable = False

    loss_value = 0
    for (batch, labels) in data_loader:
        loss_value += cross_loss(model, batch, labels) / len(batch)

    model.trainable = True
    return loss_value / len(data_loader)


def get_mean_distance(model, inputs, labels):
    outputs = model(inputs)
    predictions = np.argmax(outputs, axis=1)

    return np.sum(np.abs(predictions - labels))


def evaluate_error(model, data_loader):
    """
    Evaluate the model without back propagation.
    """
    model.trainable = False

    error_value = 0

    for (batch, labels) in data_loader:
        error_value += get_mean_distance(model, batch, labels) / len(batch)

    model.trainable = True
    return error_value / len(data_loader)


if __name__ == "__main__":
    LABELS = [9, 3]
    FULL_LABELS = create_label(LABELS, 10)

    print(FULL_LABELS)
