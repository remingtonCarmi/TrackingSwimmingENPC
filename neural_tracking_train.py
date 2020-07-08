"""
This script allows the user to train some model.
"""
# Exceptions
from src.d4_modelling_neural.loading_data.transformations.exceptions.exception_classes import FindPathDataError, PaddingError

# To train the model
from src.d4_modelling_neural_magnifier import train_magnifier

# The main module
import tensorflow as tf


# --- BEGIN : !! TO MODIFY !! --- #
TRIES = "/tries"
# Parameters for data
VIDEO_NAMES_TRAIN = ["vid0"]
VIDEO_NAMES_VALID = ["vid1"]
NUMBER_TRAINING = 3
DIMENSIONS = [110, 1820]

# Parameters for loading the data
SCALE = 35
DATA_AUGMENTING = False

# Parameters for the training
NB_EPOCHS = 8
BATCH_SIZE = 2
WINDOW_SIZE = 200
RECOVERY = 100
TRADE_OFF = 100
# --- END : !! TO MODIFY !! --- #


# Pack the variables
DATA_PARAM = [VIDEO_NAMES_TRAIN, VIDEO_NAMES_VALID, NUMBER_TRAINING, DIMENSIONS]
LOADING_PARAM = [SCALE, DATA_AUGMENTING]
TRAINING_PARAM = [NB_EPOCHS, BATCH_SIZE, WINDOW_SIZE, RECOVERY, TRADE_OFF]


"""# To avoid memory problems
TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=TF_CONFIG_)"""


# -- Verify that a GPU is used -- #
print("Is a GPU used for computations ?\n", tf.config.experimental.list_physical_devices('GPU'))

try:
    train_magnifier(DATA_PARAM, LOADING_PARAM, TRAINING_PARAM, TRIES)
except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
