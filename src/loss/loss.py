from tensorflow import GradientTape
from tensorflow import norm, convert_to_tensor
import numpy as np


def create_label(labels):
    nb_labels = len(labels)
    full_labels = np.zeros((len(labels), 10))
    for idx_label in range(nb_labels):
        full_labels[idx_label, int(labels[idx_label])] = 1

    return convert_to_tensor(full_labels, dtype=np.float32)


def evaluate(model, inputs, labels):
    full_labels = create_label(labels)
    with GradientTape() as tape:
        loss_value = cross_loss(model, inputs, full_labels)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(model, inputs, full_labels):
    outputs = model(inputs)

    return norm(outputs - full_labels) ** 2


if __name__ == "__main__":
    LABELS = [2, 3]
    FULL_LABELS = create_label(LABELS)

    print(FULL_LABELS)
