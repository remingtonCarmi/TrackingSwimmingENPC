from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data_generation.data_loader import DataLoader
from data_generation.data_generator import DataGenerator
from network_theo import EasyModel
from loss.loss import evaluate
from tensorflow.keras.optimizers import Adam


# --- Parameters --- #
# Parameters to get the data
PATH_DATA = Path("../output/test/vid1/")
PATH_LABEL = Path("../output/test/vid1.csv")
POURCENTAGE = [1, 0]

# Parameters for the training
NB_EPOCHS = 1
BATCH_SIZE = 2


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
TRAIN_SET = GENERATOR.train
TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=BATCH_SIZE)


# --- Define the MODEL --- #
MODEL = EasyModel()
OPTIMIZER = Adam()


# --- For statistics --- #
NB_BATCHES = len(TRAIN_DATA)
LOSSES = np.zeros(NB_EPOCHS * NB_BATCHES)


# --- Training --- #
for epoch in range(NB_EPOCHS):
    for (idx_batch, batch) in enumerate(TRAIN_DATA):
        (inputs, labels) = batch

        # Compute the loss and the gradients
        (loss_value, grads) = evaluate(MODEL, inputs, labels)

        # Optimize
        OPTIMIZER.apply_gradients(zip(grads, MODEL.trainable_variables))

        # Register statistics
        LOSSES[idx_batch + epoch * NB_BATCHES] = loss_value

print(MODEL(TRAIN_DATA[1][0]))
print(TRAIN_DATA[1][1])
plt.plot(LOSSES)
plt.show()
