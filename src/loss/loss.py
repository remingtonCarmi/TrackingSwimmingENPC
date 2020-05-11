from tensorflow import GradientTape
from tensorflow import math, norm, convert_to_tensor, cast
import numpy as np


def evaluate(model, inputs, labels):
    with GradientTape() as tape:
        loss_value = cross_loss(model, inputs, labels, training=True)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(model, inputs, labels, training):
    outputs = model(inputs, training=training)

    # Get the guess position of the head
    # estimated_pos = math.argmax(outputs, axis=1)
    # estimated_pos_tf = cast(estimated_pos, dtype=np.float32)

    # Transform the label
    # labels_tf = convert_to_tensor(labels)

    return outputs[0][0]  # norm(estimated_pos_tf - labels_tf, ord=2) ** 2
