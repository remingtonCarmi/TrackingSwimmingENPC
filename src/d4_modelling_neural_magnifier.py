from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# To generate the data
from src.d4_modelling_neural.loading_data.data_generator import generate_data

# To load the sets
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The models
from src.d4_modelling_neural.networks.easy_model import EasyModel

# To slice the lanes
from src.d4_modelling_neural.magnifier.slice_lanes import slice_lanes

# The loss

# The optimizer
from tensorflow.keras.optimizers import Adam

# The main module
import tensorflow as tf


# --- BEGIN : !! TO MODIFY !! --- #
# Parameters for data
VIDEO_NAMES_TRAIN = ["vid0"]
VIDEO_NAMES_VALID = ["vid1"]
FROM_COLAB = False
NUMBER_TRAINING = 0
DIMENSIONS = [110, 1820]

# For loading the data
SCALE = 35
DATA_AUGMENTING = False
# For the training
NB_EPOCHS = 5
BATCH_SIZE = 2
WINDOW_SIZE = 200
RECOVERY = 100
# --- END : !! TO MODIFY !! --- #


"""# To avoid memory problems
TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=TF_CONFIG_)"""


# -- Verify that a GPU is used -- #
print("Is a GPU used for computations ?\n", tf.config.experimental.list_physical_devices('GPU'))


# -- Paths to the data -- #
if FROM_COLAB:
    PATH_BEGIN = ""
else:
    PATH_BEGIN = "../"

PATHS_LABEL_TRAIN = []
for video_name_train in VIDEO_NAMES_TRAIN:
    PATHS_LABEL_TRAIN.append(Path(PATH_BEGIN + "data/2_processed_positions/tries/{}.csv".format(video_name_train)))

PATHS_LABEL_VALID = []
for video_name_valid in VIDEO_NAMES_VALID:
    PATHS_LABEL_VALID.append(Path(PATH_BEGIN + "data/2_processed_positions/tries/{}.csv".format(video_name_valid)))

STARTING_DATA_PATHS = Path(PATH_BEGIN + "data/1_intermediate_top_down_lanes/lanes/tries")
STARTING_CALIBRATION_PATHS = Path(PATH_BEGIN + "data/1_intermediate_top_down_lanes/calibration/tries")


# --- Generate and load the sets--- #
TRAIN_DATA = generate_data(PATHS_LABEL_TRAIN, STARTING_DATA_PATHS, STARTING_CALIBRATION_PATHS)
VALID_DATA = generate_data(PATHS_LABEL_VALID, STARTING_DATA_PATHS, STARTING_CALIBRATION_PATHS)

TRAIN_SET = DataLoader(TRAIN_DATA, batch_size=BATCH_SIZE, scale=SCALE, dimensions=DIMENSIONS, data_augmenting=DATA_AUGMENTING)
VALID_SET = DataLoader(VALID_DATA, batch_size=BATCH_SIZE, scale=SCALE, dimensions=DIMENSIONS, data_augmenting=DATA_AUGMENTING)
print("The training set is composed of {} image".format(len(TRAIN_SET)))
print("The validation set is composed of {} image".format(len(VALID_SET)))

# --- Define the MODEL --- #
MODEL = EasyModel()

# Get the weights of the previous trainings
PATH_WEIGHT = Path(PATH_BEGIN + "data/3_models_weights")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build()
    PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}.h5".format(0, NUMBER_TRAINING - 1)
else:
    PATH_FORMER_TRAINING = PATH_WEIGHT / "hard_model_nb_classes_{}_{}.h5".format(0, NUMBER_TRAINING - 1)

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
    for (idx_batch, batch) in enumerate(TRAIN_SET):
        (lanes, labels) = batch

        # Get the sub images
        (sub_lanes, sub_labels) = slice_lanes(lanes, labels, WINDOW_SIZE, RECOVERY)

        # Compute the loss and the gradients
        (loss_value, grads) = get_loss(MODEL, lanes, labels)

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
PATH_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING)
MODEL.save_weights(str(PATH_TRAINING))


# To save the plots
PATH_SAVE_FIG = Path(PATH_BEGIN + "reports/figures_results/")
PATH_SAVE_LOSS = PATH_SAVE_FIG / "loss_easy_model_nb_classes_{}_{}.jpg".format(0, NUMBER_TRAINING)
PATH_SAVE_ACCURACY = PATH_SAVE_FIG / "mean_error_easy_model_nb_classes_{}_{}.jpg".format(0, NUMBER_TRAINING)


# Observe results
MODEL.trainable = False
for (idx_batch, batch) in enumerate(TRAIN_DATA):
    (inputs, labels) = batch
    PREDICTIONS = MODEL(inputs)
    print(PREDICTIONS)
    print("Predictions", np.argmax(PREDICTIONS, axis=1))
    print("label", labels)


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
