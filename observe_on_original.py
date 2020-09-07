"""
This script allows the user to observe the behavior of the trained models on the original video clip.
"""
from pathlib import Path

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import (
    FindPathDataError,
    PaddingError,
)
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError

# To compute the predictions
from src.d5_model_evaluation_magnifier import evaluate_model

# To observe the model
from src.d7_visualization_predictions import observe_model

# To make a video from images
from src.d0_utils.store_load_data.make_video import make_video


# --- BEGIN : !! TO MODIFY !! --- #
REAL_RUN = True
# For the data
VIDEO_NAME = "vid0"
LANE_NUMBER = -1
DIMENSIONS = [108, 1820]
SCALE = 35
BEGIN_TIME = 14  # 10
END_TIME = 16  # 36

# For the models
DEEP_MODELS = [True, True]
NUMBER_TRAININGS = [4, 1]
NB_EPOCHS = [1, 22]
BATCH_SIZES = [12, 12]
WINDOW_SIZES = [150, 30]
RECOVERIES = [75, 29]
# --- END : !! TO MODIFY !! --- #


# --- Set the parameters --- #
if REAL_RUN:
    TRIES = ""
else:
    TRIES = "/tries"
if DEEP_MODELS[0]:
    MODEL_TYPE1 = "/deep_model"
else:
    MODEL_TYPE1 = "/simple_model"
if DEEP_MODELS[1]:
    MODEL_TYPE2 = "/deep_model"
else:
    MODEL_TYPE2 = "/simple_model"


# Pack the variables
DATA_PARAM = [VIDEO_NAME, LANE_NUMBER, DIMENSIONS, SCALE, BEGIN_TIME, END_TIME]
MODELS_PARAM = [MODEL_TYPE1, MODEL_TYPE2, NUMBER_TRAININGS, NB_EPOCHS, BATCH_SIZES, WINDOW_SIZES, RECOVERIES]


try:
    # --- Get the predictions --- #
    PREDICTION_MEMORIES = observe_model(DATA_PARAM, MODELS_PARAM, evaluate_model, TRIES)

    # --- Make the video --- #
    print("Making the video...")
    DESTINATION_VIDEO = Path("data/5_model_output/videos{}".format(TRIES))
    NAME_PREDICTED_VIDEO = "predicted_original_{}_{}_window_{}_{}_window_{}.mp4".format(
        VIDEO_NAME, MODEL_TYPE1[1:], WINDOW_SIZES[0], MODEL_TYPE2[1:], WINDOW_SIZES[1]
    )
    make_video(
        NAME_PREDICTED_VIDEO,
        PREDICTION_MEMORIES.get_original_frames(),
        fps=PREDICTION_MEMORIES.fps,
        destination=DESTINATION_VIDEO,
    )

except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
except AlreadyExistError as already_exist_error:
    print(already_exist_error.__repr__())
except FindPathError as find_path_error:
    print(find_path_error.__repr__())
