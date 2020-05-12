from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator
from src.network_theo import EasyModel
from src.loss.loss import evaluate
from tensorflow.keras.optimizers import Adam


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid1"
POURCENTAGE = [1, 0]  # [Training set, Validation set]

# Parameters for the training
EASY_MODEL = True
NUMBER_TRAINING = 2
NB_EPOCHS = 1
BATCH_SIZE = 2


# --- Parameters --- #
# Parameters to get the data
PATH_DATA = Path("../output/test/{}/".format(VIDEO_NAME))
PATH_LABEL = Path("../output/test/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
TRAIN_SET = GENERATOR.train
TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=BATCH_SIZE)


# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel()
else:
    MODEL = EasyModel()
MODEL.build(TRAIN_DATA[0][0].shape)
# Get the weights of the previous trainings
PATH_WEIGHT = Path("../trained_weights/")
if NUMBER_TRAINING > 0:
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING - 1)
    MODEL.load_weights(str(PATH_FORMER_TRAINING))
# Optimizer
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

# --- Save the weights --- #
if EASY_MODEL:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING)
else:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING)
MODEL.save_weights(str(PATH_TRAINING))


plt.plot(LOSSES)
plt.show()
