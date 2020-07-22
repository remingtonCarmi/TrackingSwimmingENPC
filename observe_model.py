"""
This script allows the user to observe the behavior of the trained models.
"""
# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import FindPathDataError, PaddingError
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError

# To compute the predictions
from src.d5_model_evaluation_magnifier import evaluate_model

# To observe the model
from src.d7_visualization_magnifier import observe_models


# --- BEGIN : !! TO MODIFY !! --- #
REAL_RUN = False
# For the data
VIDEO_NAME = "vid0"
LANE_NUMBER = 1
DIMENSIONS = [108, 1820]
SCALE = 35

# For the models
DEEP_MODELS = [False, False]
NUMBER_TRAININGS = [1, 1]
NB_EPOCHS = [12, 15]
BATCH_SIZES = [12, 14]
WINDOW_SIZES = [150, 30]
RECOVERIES = [75, 29]
# --- END : !! TO MODIFY !! --- #


# --- Set the parameters --- #
if REAL_RUN:
    TRIES = ""
else:
    TRIES = "/tries"
if DEEP_MODELS[0]:
    MODEL_TYPE1 = "/rough_deep_model"
else:
    MODEL_TYPE1 = "/rough_simple_model"
if DEEP_MODELS[1]:
    MODEL_TYPE2 = "/tight_deep_model"
else:
    MODEL_TYPE2 = "/tight_simple_model"


# Pack the variables
DATA_PARAM = [VIDEO_NAME, LANE_NUMBER, DIMENSIONS, SCALE]
MODELS_PARAM = [MODEL_TYPE1, MODEL_TYPE2, NUMBER_TRAININGS, NB_EPOCHS, BATCH_SIZES, WINDOW_SIZES, RECOVERIES]


try:
    observe_models(DATA_PARAM, MODELS_PARAM, evaluate_model, TRIES)
except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
except AlreadyExistError as already_exist_error:
    print(already_exist_error.__repr__())
except FindPathError as find_path_error:
    print(find_path_error.__repr__())
