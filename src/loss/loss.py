from tensorflow import GradientTape
import numpy as np


def evaluate(model, inputs, labels):
    size_batch = len(inputs)

    with GradientTape() as tape:
        # Compute the outputs
        outputs = model(inputs)
        loss_value = 0
        # We compute the loss for each element of the batch
        for idx in range(size_batch):
            loss_value += cross_loss(outputs[idx], labels[idx])

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def cross_loss(guess, label):
    # Get the guess position of the head
    estimated_pos = np.argmax(guess)

    return (label - estimated_pos) ** 2
