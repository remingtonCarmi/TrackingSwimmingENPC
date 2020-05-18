from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator
from src.networks.easy_model import EasyModel
from src.networks.hard_model import HardModel
from src.loss.loss import get_loss, evaluate_loss, evaluate_accuracy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid1"
PERCENTAGE = [0.8, 0.2]  # [Training set, Validation set]
FROM_COLAB = True

# Parameters for the training
NUMBER_TRAINING = 0
EASY_MODEL = True
NB_EPOCHS = 20
BATCH_SIZE = 2


# -- Verify that a GPU is used -- #
print("Is a GPU used for computations ?\n", tf.config.experimental.list_physical_devices('GPU'))


# --- Parameters --- #
# Parameters to get the data
if FROM_COLAB:
    PATH_BEGIN = ""
else:
    PATH_BEGIN = "../"
PATH_DATA = Path(PATH_BEGIN + "output/test/{}/".format(VIDEO_NAME))
PATH_LABEL = Path(PATH_BEGIN + "output/test/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
TRAIN_SET = GENERATOR.train
VAL_SET = GENERATOR.valid
TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=BATCH_SIZE)
VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=len(VAL_SET), for_train=False)
(VALID_SAMPLES, VALID_LABELS) = VALID_DATA[0]
print("The training set is composed of {} images".format(len(TRAIN_SET)))
print("The validation set is composed of {} images".format(len(VALID_SAMPLES)))

# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel()
else:
    MODEL = HardModel()
# Get the weights of the previous trainings
PATH_WEIGHT = Path(PATH_BEGIN + "trained_weights/")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build(TRAIN_DATA[0][0].shape)
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "hard_model_{}.h5".format(NUMBER_TRAINING - 1)
    # Load the weights
    MODEL.load_weights(str(PATH_FORMER_TRAINING))
# Optimizer
OPTIMIZER = Adam()


# --- For statistics --- #
LOSSES_ON_TRAIN = np.zeros(NB_EPOCHS)
LOSSES_ON_VAL = np.zeros(NB_EPOCHS)
ACCURACIES_ON_TRAIN = np.zeros(NB_EPOCHS)
ACCURACIES_ON_VAL = np.zeros(NB_EPOCHS)


# --- Training --- #
for epoch in range(NB_EPOCHS):
    sum_loss = 0
    sum_accuracy = 0
    TRAIN_DATA.on_epoch_end()
    for (idx_batch, batch) in enumerate(TRAIN_DATA):
        (inputs, labels) = batch

        # Compute the loss and the gradients
        (loss_value, grads) = get_loss(MODEL, inputs, labels)

        # Optimize
        OPTIMIZER.apply_gradients(zip(grads, MODEL.trainable_variables))

        # Register statistics
        sum_loss += loss_value / len(labels)
        sum_accuracy += evaluate_accuracy(MODEL, inputs, labels)

    # Register the loss on train
    LOSSES_ON_TRAIN[epoch] = sum_loss / len(TRAIN_DATA)
    # Register the loss on val
    LOSSES_ON_VAL[epoch] = evaluate_loss(MODEL, VALID_SAMPLES, VALID_LABELS) / len(VALID_SAMPLES)
    # Register the accuracy on train
    ACCURACIES_ON_TRAIN[epoch] = sum_accuracy / len(TRAIN_DATA)
    # Register the accuracy on val
    ACCURACIES_ON_VAL[epoch] = evaluate_accuracy(MODEL, VALID_SAMPLES, VALID_LABELS) / len(VALID_SAMPLES)


# --- Save the weights --- #
if EASY_MODEL:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_{}.h5".format(NUMBER_TRAINING)
else:
    PATH_TRAINING = PATH_WEIGHT / "hard_model_{}.h5".format(NUMBER_TRAINING)
MODEL.save_weights(str(PATH_TRAINING))


plt.plot(LOSSES_ON_TRAIN, label="Loss on train set")
plt.plot(LOSSES_ON_VAL, label="Loss on validation set")
plt.xlabel("Number of epoch")
plt.legend()

plt.figure()
plt.plot(ACCURACIES_ON_TRAIN, label="Accuracy on train set")
plt.plot(ACCURACIES_ON_VAL, label="Accuracy on validation set")
plt.xlabel("Number of epoch")
plt.legend()
plt.show()
