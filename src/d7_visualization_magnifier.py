"""
This script creates a video where the predicted LABELS are printed on the LANES.
"""
from pathlib import Path
import numpy as np

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import FindPathDataError, PaddingError
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError

# To manage the graphic
from src.d7_visualization.graphic_manager import GraphicManager

# To manage the video
from src.d7_visualization.video_manager import VideoManager

# To generate and load data
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The model
from src.d4_modelling_neural.magnifier.zoom_model import ZoomModel
from src.d4_modelling_neural.magnifier.zoom_model_deep import ZoomModelDeep

# To slice the lanes
from src.d4_modelling_neural.magnifier.slice_sample_lane.image_objects.image_magnifier import ImageMagnifier
from src.d4_modelling_neural.magnifier.slice_sample_lane.slice_lanes import slice_lanes

# To merge the predictions
from src.d7_visualization.compute_prediction import merge_predictions

# To make a video from images
from src.d0_utils.store_load_data.make_video import make_video


# --- BEGIN : !! TO MODIFY !! --- #
REAL_RUN = False
# For the data
VIDEO_NAME = "vid0"
LANE_NUMBER = 1
DIMENSIONS = [108, 1820]
SCALE = 35

# For the MODEL
DEEP_MODEL = True
NUMBER_TRAINING = 1
WINDOW_SIZE = 150
RECOVERY = 100
# --- END : !! TO MODIFY !! --- #

# --- Set the parameters --- #
if REAL_RUN:
    TRIES = ""
else:
    TRIES = "/tries"
if DEEP_MODEL:
    MODEL_TYPE = "/deep_model"
else:
    MODEL_TYPE = "/simple_model"


# --- Set the paths --- #
PATH_VIDEO = Path("../data/1_raw_videos/{}.mp4".format(VIDEO_NAME))
PATH_LABEL = [Path("../data/3_processed_positions{}/{}.csv".format(TRIES, VIDEO_NAME))]
STARTING_DATA_PATH = Path("../data/2_intermediate_top_down_lanes/lanes{}".format(TRIES))
STARTING_CALIBRATION_PATH = Path("../data/2_intermediate_top_down_lanes/calibration{}".format(TRIES))
PATH_SAVE_GRAPHIC = Path("../reports/graphic_results{}/{}.jpg".format(TRIES, VIDEO_NAME))

PATH_WEIGHT = Path("../data/4_models_weights{}/magnifier{}".format(TRIES, MODEL_TYPE))
PATH_CURRENT_WEIGHT = PATH_WEIGHT / "window_{}_epoch_{}_batch_{}_{}.h5".format(WINDOW_SIZE, 15, 3, NUMBER_TRAINING)


try:
    # --- Generate and load the sets --- #
    DATA = generate_data(PATH_LABEL, STARTING_DATA_PATH, STARTING_CALIBRATION_PATH, take_all=False, lane_number=LANE_NUMBER)
    SET = DataLoader(DATA, scale=SCALE, batch_size=1, dimensions=DIMENSIONS, standardization=True, augmentation=False, flip=True)
    SET_VISU = DataLoader(DATA, scale=SCALE, batch_size=1, dimensions=DIMENSIONS, standardization=False, augmentation=False, flip=False)

    print("The set is composed of {} images".format(len(DATA)))

    # --- Define the graphic manager --- #
    GRAPHIC_MANAGER = GraphicManager(PATH_VIDEO, STARTING_CALIBRATION_PATH / "{}.txt".format(VIDEO_NAME), SCALE, len(SET), DIMENSIONS[1])

    # --- Define the video manager --- #
    VIDEO_MANAGER = VideoManager(len(SET))

    # --- Define the MODEL --- #
    if DEEP_MODEL:
        MODEL = ZoomModelDeep()
    else:
        MODEL = ZoomModel()

    # --- Get the weights of the trainings --- #
    # Build the model to load the weights
    (LANES, LABELS) = SET[0]
    (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)
    MODEL.build(SUB_LANES.shape)
    # Load the weights
    MODEL.load_weights(str(PATH_CURRENT_WEIGHT))

    # --- Evaluate the set --- #
    MODEL.trainable = False

    for (idx_batch, batch) in enumerate(SET):
        (LANES, LABELS) = batch
        SWIMMING_WAY = DATA[idx_batch, 3]

        # -- Get the predictions -- #
        # Get the sub-images with the image that has been standardized
        (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)
        # Compute predictions
        PREDICTIONS = MODEL(SUB_LANES)[:: int(SWIMMING_WAY)]

        # -- Merge the predictions -- #
        # Get the original lane_magnifier
        MAGNIFIER_ORIGINAL_IMAGE = ImageMagnifier(SET_VISU[idx_batch][0][0], SET_VISU[idx_batch][1][0], WINDOW_SIZE, RECOVERY)
        # The magnifier original image is only used to get the limits of the windows
        INDEX_PREDS = merge_predictions(MAGNIFIER_ORIGINAL_IMAGE, PREDICTIONS)
        print(INDEX_PREDS)
        # -- For the graphic -- #
        FRAME_NAME = DATA[idx_batch, 0].parts[-1][: -4]
        if SWIMMING_WAY == 1:
            LABEL = LABELS[0, 1]
        else:
            LABEL = DIMENSIONS[1] - LABELS[0, 1]
        GRAPHIC_MANAGER.update(idx_batch, FRAME_NAME, INDEX_PREDS, LABEL)

        # -- For the video -- #
        VIDEO_MANAGER.update(idx_batch, MAGNIFIER_ORIGINAL_IMAGE.lane, INDEX_PREDS)

    # --- Make the graphic --- #
    GRAPHIC_MANAGER.make_graphic(PATH_SAVE_GRAPHIC)

    # --- Make the video --- #
    print("Making the video...")
    DESTINATION_VIDEO = Path("../data/5_model_output/videos{}".format(TRIES))
    NAME_PREDICTED_VIDEO = "predicted_{}_window_{}_recovery_{}{}_{}.mp4".format(VIDEO_NAME, WINDOW_SIZE, RECOVERY, "_" + MODEL_TYPE[1:], NUMBER_TRAINING)
    make_video(NAME_PREDICTED_VIDEO, VIDEO_MANAGER.lanes_with_preds, destination=DESTINATION_VIDEO)

except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
except AlreadyExistError as already_exist_error:
    print(already_exist_error.__repr__())
except FindPathError as find_path_error:
    print(find_path_error.__repr__())
