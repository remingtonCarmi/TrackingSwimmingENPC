from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator
from src.networks.easy_model import EasyModel
from src.loss.loss import get_loss, evaluate
from tensorflow.keras.optimizers import Adam


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid1"
POURCENTAGE = [0.8, 0.2]  # [Training set, Validation set]

# Parameters for the training
NUMBER_TRAINING = 0
EASY_MODEL = True
NB_EPOCHS = 5
BATCH_SIZE = 2


# --- Parameters --- #
# Parameters to get the data
PATH_DATA = Path("../output/test/{}/".format(VIDEO_NAME))
PATH_LABEL = Path("../output/test/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
TRAIN_SET = GENERATOR.train
VAL_SET = GENERATOR.valid
TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=BATCH_SIZE)
VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=len(VAL_SET))
(VALID_SAMPLES, VALID_LABELS) = VALID_DATA[0]


# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel()
else:
    MODEL = EasyModel()
# Get the weights of the previous trainings
PATH_WEIGHT = Path("../trained_weights/")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build(TRAIN_DATA[0][0].shape)
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING - 1)
    # Load the weights
    MODEL.load_weights(str(PATH_FORMER_TRAINING))
# Optimizer
OPTIMIZER = Adam()


# --- For statistics --- #
NB_BATCHES = len(TRAIN_DATA)
LOSSES_ON_TRAIN = np.zeros(NB_EPOCHS)
LOSSES_ON_VAL = np.zeros(NB_EPOCHS)


# --- Training --- #
for epoch in range(NB_EPOCHS):
    sum_loss = 0
    for (idx_batch, batch) in enumerate(TRAIN_DATA):
        (inputs, labels) = batch

        # Compute the loss and the gradients
        (loss_value, grads) = get_loss(MODEL, inputs, labels)

        # Optimize
        OPTIMIZER.apply_gradients(zip(grads, MODEL.trainable_variables))

        # Register statistics
        sum_loss += loss_value
    # Register the loss on train
    LOSSES_ON_TRAIN[epoch] = sum_loss
    # Register the loss on val
    LOSSES_ON_VAL[epoch] = evaluate(MODEL, VALID_SAMPLES, VALID_LABELS)

# --- Save the weights --- #
if EASY_MODEL:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING)
else:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING)
MODEL.save_weights(str(PATH_TRAINING))


plt.plot(LOSSES_ON_TRAIN, label="Loss on train set")
plt.plot(LOSSES_ON_TRAIN, label="Loss on validation set")
plt.legend()
plt.show()
