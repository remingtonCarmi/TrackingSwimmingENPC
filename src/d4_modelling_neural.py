from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.d4_modelling_neural.data_generation import DataLoader
from src.d4_modelling_neural.data_generation import DataGenerator
from src.d4_modelling_neural.networks import EasyModel
from src.d4_modelling_neural.networks import HardModel
from src.d4_modelling_neural.loss import get_loss, evaluate_loss, evaluate_error, get_mean_distance
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid0"
PERCENTAGE = 0.8959  # percentage of the training set
FROM_COLAB = False
NB_CLASSES = 10


# To avoid memory problems
TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=TF_CONFIG_)


# Parameters for the training
NUMBER_TRAINING = 0
EASY_MODEL = True
NB_EPOCHS = 5
BATCH_SIZE = 6
DATA_AUGMENTING = False

# -- Verify that a GPU is used -- #
print("Is a GPU used for computations ?\n", tf.config.experimental.list_physical_devices('GPU'))


# --- Parameters --- #
# Parameters to get the data
if FROM_COLAB:
    PATH_BEGIN = ""
else:
    PATH_BEGIN = "../"
PATH_DATA = Path(PATH_BEGIN + "output/tries/{}/".format(VIDEO_NAME))
PATH_LABEL = Path(PATH_BEGIN + "output/tries/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
TRAIN_SET = GENERATOR.train
VAL_SET = GENERATOR.valid
TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=BATCH_SIZE, data_augmenting=DATA_AUGMENTING, nb_classes=NB_CLASSES)
VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=1, nb_classes=NB_CLASSES)
print("The training set is composed of {} images".format(len(TRAIN_SET)))
print("The validation set is composed of {} images".format(len(VAL_SET)))

# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel(NB_CLASSES)
else:
    MODEL = HardModel(NB_CLASSES)
# Get the weights of the previous trainings
PATH_WEIGHT = Path(PATH_BEGIN + "trained_weights/")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build(TRAIN_DATA[0][0].shape)
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "hard_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING - 1)
    # Load the weights
    MODEL.load_weights(str(PATH_FORMER_TRAINING))
# Optimizer
OPTIMIZER = Adam()


# --- For statistics --- #
LOSSES_ON_TRAIN = np.zeros(NB_EPOCHS)
LOSSES_ON_VAL = np.zeros(NB_EPOCHS)
ERRORS_ON_TRAIN = np.zeros(NB_EPOCHS)
ERRORS_ON_VAL = np.zeros(NB_EPOCHS)


# --- Training --- #
for epoch in range(NB_EPOCHS):
    sum_loss = 0
    sum_errors = 0
    for (idx_batch, batch) in enumerate(TRAIN_DATA):
        (inputs, labels) = batch

        # Compute the loss and the gradients
        (loss_value, grads) = get_loss(MODEL, inputs, labels)

        # Optimize
        OPTIMIZER.apply_gradients(zip(grads, MODEL.trainable_variables))

        # Register statistics
        sum_loss += loss_value / len(labels)
        sum_errors += get_mean_distance(MODEL, inputs, labels) / len(labels)

    # Register the loss on train
    LOSSES_ON_TRAIN[epoch] = sum_loss / len(TRAIN_DATA)
    # Register the loss on val
    LOSSES_ON_VAL[epoch] = evaluate_loss(MODEL, VALID_DATA)
    # Register the accuracy on train
    ERRORS_ON_TRAIN[epoch] = sum_errors / len(TRAIN_DATA)
    # Register the accuracy on val
    ERRORS_ON_VAL[epoch] = evaluate_error(MODEL, VALID_DATA)

    # Shuffle data
    TRAIN_DATA.on_epoch_end()


# --- Save the weights --- #
if EASY_MODEL:
    PATH_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING)
else:
    PATH_TRAINING = PATH_WEIGHT / "hard_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING)
MODEL.save_weights(str(PATH_TRAINING))


# To save the plots
PATH_SAVE_FIG = Path(PATH_BEGIN + "output/model_stats/")
if EASY_MODEL:
    PATH_SAVE_LOSS = PATH_SAVE_FIG / "loss_easy_model_nb_classes_{}_{}.jpg".format(NB_CLASSES, NUMBER_TRAINING)
    PATH_SAVE_ACCURACY = PATH_SAVE_FIG / "mean_error_easy_model_nb_classes_{}_{}.jpg".format(NB_CLASSES, NUMBER_TRAINING)

else:
    PATH_SAVE_LOSS = PATH_SAVE_FIG / "loss_hard_model_nb_classes_{}_{}.jpg".format(NB_CLASSES, NUMBER_TRAINING)
    PATH_SAVE_ACCURACY = PATH_SAVE_FIG / "mean_error_hard_model_nb_classes_{}_{}.jpg".format(NB_CLASSES, NUMBER_TRAINING)


# Observe results
MODEL.trainable = False
for (idx_batch, batch) in enumerate(TRAIN_DATA):
    (inputs, labels) = batch
    PREDICTIONS = MODEL(inputs)
    print(PREDICTIONS)
    print("Predictions", np.argmax(PREDICTIONS, axis=1))
    print("labels", labels)


# Plot the results
plt.plot(LOSSES_ON_TRAIN, label="Loss on train set")
plt.plot(LOSSES_ON_VAL, label="Loss on validation set")
plt.xlabel("Number of epoch")
plt.legend()
plt.savefig(PATH_SAVE_LOSS)
plt.show()
plt.close()


plt.plot(ERRORS_ON_TRAIN, label="Mean error on train set")
plt.plot(ERRORS_ON_VAL, label="Mean error on validation set")
plt.xlabel("Number of epoch")
plt.legend()
plt.savefig(PATH_SAVE_ACCURACY)
plt.show()
