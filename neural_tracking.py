""" This script create a video a one particular line with the prediction of the neural network."""

from pathlib import Path
import numpy as np
from src.d4_modelling_neural.loading_data import DataLoader
from src.d4_modelling_neural.loading_data import DataGenerator
from src.d4_modelling_neural.networks import EasyModel
from src.d4_modelling_neural.networks import HardModel
from src.d7_visualisation.visualization_deep import animation_one_lane
from src.d0_utils.store_load_matrix.make_video import make_video
from src.d0_utils.store_load_matrix import AlreadyExistError, FindErrorStore


# Parameters for data
VIDEO_NAME = "vid0"
PERCENTAGE = 0.875  # percentage of the training set

NB_CLASSES = 10

# Parameters for the training
NUMBER_TRAINING = 2
EASY_MODEL = True

# To build a visualization video
CREATE_VIDEO = True


# --- Parameters --- #
# Parameters to get the data
PATH_DATA = Path("data/lanes/{}/".format(VIDEO_NAME))
PATH_LABEL = Path("data/head_points/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE, for_visu=True)
VAL_SET = GENERATOR.valid

VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=1, nb_classes=NB_CLASSES)
print("The validation set is composed of {} images".format(len(VAL_SET)))


# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel(NB_CLASSES)
else:
    MODEL = HardModel(NB_CLASSES)
# Get the weights of the previous trainings
PATH_WEIGHT = Path("trained_weights/")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build(VALID_DATA[0][0].shape)
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}_trained.h5".format(NB_CLASSES,
                                                                                             NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "hard_model_nb_classes_{}_{}_trained_second.h5".format(NB_CLASSES,
                                                                                                    NUMBER_TRAINING - 1)
    # Load the weights
    MODEL.load_weights(str(PATH_FORMER_TRAINING))


# --- Evaluation --- #
MODEL.trainable = False
PREDICTIONS = np.zeros(len(VALID_DATA))
VALID_LABELS = np.zeros(len(VALID_DATA))

for (idx, (batch, label)) in enumerate(VALID_DATA):
    OUTPUTS = MODEL(batch)
    PREDICTIONS[idx] = np.argmax(OUTPUTS, axis=1)
    VALID_LABELS[idx] = int(label)

print("Predictions")
print(PREDICTIONS)
print("Labels")
print(VALID_LABELS)

# --- Visualisation --- #

# FRAMES = visualize(VAL_SET[:, 0], PREDICTIONS, PATH_DATA, NB_CLASSES)
FRAMES = animation_one_lane(VAL_SET[:, 0], PREDICTIONS, PATH_DATA, NB_CLASSES)

if CREATE_VIDEO:
    try:
        make_video("prediction_" + VIDEO_NAME + ".mp4", FRAMES[1:], destination=Path("output/videos/"))

    except AlreadyExistError as already_exists:
        print(already_exists.__repr__())

    except FindErrorStore as find_error:
        print(find_error.__repr__())
