import numpy as np
from tensorflow import GradientTape
from tensorflow import norm, convert_to_tensor, reduce_sum
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import mean_squared_error


def create_label(labels, nb_classes):
    """
    Transform the labels' batch that are integers to list of zeros where there is a one in the right place.
    """
    nb_labels = len(labels)
    full_labels = np.zeros((len(labels), nb_classes))
    for idx_label in range(nb_labels):
        full_labels[idx_label, int(labels[idx_label])] = 1

    return convert_to_tensor(full_labels, dtype=np.float32)


def create_smooth_label(labels, nb_classes, factor=0.8):
    """
    Transform the labels' batch that are integers to list of smoothed labels, the higher number in the true class."
    """
    nb_labels = len(labels)
    full_labels = np.zeros((len(labels), nb_classes))
    for idx_label in range(nb_labels):
        label_class = int(labels[idx_label])
        if label_class == nb_classes - 1:
            full_labels[idx_label, label_class - 1] = 1 - factor
        elif label_class == 0:
            full_labels[idx_label, label_class + 1] = 1 - factor
        else:
            full_labels[idx_label, label_class - 1] = (1 - factor) / 2
            full_labels[idx_label, label_class + 1] = (1 - factor) / 2
        full_labels[idx_label, label_class] = factor

    return convert_to_tensor(full_labels, dtype=np.float32)


def get_loss(model, inputs, labels, nb_classes):
    """
    Get the loss and the gradient of every layer.
    """
    full_labels = create_smooth_label(labels, nb_classes)
    with GradientTape() as tape:
        loss_value = cross_loss(model, inputs, full_labels)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(model, inputs, full_labels):
    """
    Get the loss of the model.
    """
    outputs = model(inputs)

    return reduce_sum(binary_crossentropy(outputs, full_labels))


def evaluate_loss(model, inputs, labels, nb_classes):
    """
    Evaluate the model without back propagation.
    """
    model.trainable = False
    full_labels = create_label(labels, nb_classes)

    loss_value = cross_loss(model, inputs, full_labels)

    model.trainable = True
    return loss_value


def get_accuracy(model, inputs, labels):
    outputs = model(inputs)
    predictions = np.argmax(outputs, axis=1)

    return np.mean(predictions == labels)


def get_mean_distance(model, inputs, labels):
    outputs = model(inputs)
    predictions = np.argmax(outputs, axis=1)

    return np.mean(np.abs(predictions - labels))


def evaluate_accuracy(model, inputs, labels):
    """
    Evaluate the model without back propagation.
    """
    model.trainable = False

    acc_value = get_mean_distance(model, inputs, labels)

    model.trainable = True
    return acc_value


if __name__ == "__main__":
    LABELS = [9, 3]
    FULL_LABELS = create_smooth_label(LABELS, 10)

    print(FULL_LABELS)
