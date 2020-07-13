"""
This script creates a video where the predicted LABELS are printed on the LANES.
"""
from pathlib import Path

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions import FindPathDataError, PaddingError
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError

# To generate and load data
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The MODEL
from src.d4_modelling_neural.magnifier.zoom_model import ZoomModel

# To slice the LANES
from src.d4_modelling_neural.magnifier.slice_lane.image_magnifier.image_magnifier import ImageMagnifier
from src.d4_modelling_neural.magnifier.slice_lane.slice_lanes import slice_lanes

# To add the PREDICTIONS to the LANES
from src.d7_visualization.add_prediction_magnifier import add_prediction

# To make a video from images
from src.d0_utils.store_load_data.make_video import make_video


# --- BEGIN : !! TO MODIFY !! --- #
REAL_RUN = False
# For the data
VIDEO_NAME = "vid1"
DIMENSIONS = [110, 1820]
SCALE = 35

# For the MODEL
NUMBER_TRAINING = 2
WINDOW_SIZE = 200
RECOVERY = 10
# --- END : !! TO MODIFY !! --- #

# --- Set the parameter --- #
if REAL_RUN:
    TRIES = ""
else:
    TRIES = "/tries"


# --- Set the paths --- #
PATH_LABEL = [Path("../data/2_processed_positions{}/{}.csv".format(TRIES, VIDEO_NAME))]
STARTING_DATA_PATH = Path("../data/1_intermediate_top_down_lanes/LANES{}".format(TRIES))
STARTING_CALIBRATION_PATH = Path("../data/1_intermediate_top_down_lanes/calibration{}".format(TRIES))

PATH_WEIGHT = Path("../data/3_models_weights{}/magnifier".format(TRIES))


try:
    # --- Generate and load the sets --- #
    DATA = generate_data(PATH_LABEL, STARTING_DATA_PATH, STARTING_CALIBRATION_PATH, take_all=False)
    SET = DataLoader(DATA, scale=SCALE, batch_size=1, dimensions=DIMENSIONS)
    SET_VISU = DataLoader(DATA, scale=SCALE, batch_size=1, dimensions=DIMENSIONS, standardization=False)

    print("The set is composed of {} images".format(len(DATA)))

    # --- Define the MODEL --- #
    MODEL = ZoomModel()

    # - Get the weights of the trainings - #
    # Get the input shape to build the MODEL
    # Build the MODEL to load the weights
    (LANES, LABELS) = SET[0]
    # Get the sub images
    (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)

    # Build the MODEL
    MODEL.build(SUB_LANES.shape)

    PATH_CURRENT_WEIGHT = PATH_WEIGHT / "window_{}_{}.h5".format(WINDOW_SIZE, NUMBER_TRAINING)

    # Load the weights
    MODEL.load_weights(str(PATH_CURRENT_WEIGHT))

    # --- Evaluate the set --- #
    MODEL.trainable = False

    # Will contain all the LANES with the prediction
    LANES_PREDICTIONS = [0] * len(SET)

    for (idx_batch, batch) in enumerate(SET):
        (LANES, LABELS) = batch

        # -- Get the PREDICTIONS -- #
        # Get the sub-images with the image that has been standardized
        (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)

        # Compute PREDICTIONS
        PREDICTIONS = MODEL(SUB_LANES)

        # -- Get the original lane_magnifier -- #
        MAGNIFIER_ORIGINAL_IMAGE = ImageMagnifier(SET_VISU[idx_batch][0][0], SET_VISU[idx_batch][1][0], WINDOW_SIZE, RECOVERY)

        # -- Add the PREDICTIONS to the lane_magnifier -- #
        # Plot the prediction on the image that has NOT been standardized
        LANE_PRED = add_prediction(MAGNIFIER_ORIGINAL_IMAGE, PREDICTIONS)

        # Add the main list
        LANES_PREDICTIONS[idx_batch] = LANE_PRED

    # --- Make the video --- #
    print("Making the video...")
    DESTINATION_VIDEO = Path("../data/4_model_output/videos{}".format(TRIES))
    NAME_PREDICTED_VIDEO = "predicted_{}_recovery_{}_{}.mp4".format(VIDEO_NAME, RECOVERY, NUMBER_TRAINING)
    make_video(NAME_PREDICTED_VIDEO, LANES_PREDICTIONS, destination=DESTINATION_VIDEO)

except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
except AlreadyExistError as already_exist_error:
    print(already_exist_error.__repr__())
except FindPathError as find_path_error:
    print(find_path_error.__repr__())
