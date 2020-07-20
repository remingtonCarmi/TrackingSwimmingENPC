"""
This script creates a video where the LABELS are printed on the LANES.
"""
from pathlib import Path
import numpy as np

# Exceptions
from src.d4_modelling_neural.loading_data.transformations.tools.exceptions.exception_classes import FindPathDataError, PaddingError
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError, FindPathError

# To generate and load data
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# To make a video from images
from src.d0_utils.store_load_data.make_video import make_video


# --- BEGIN : !! TO MODIFY !! --- #
REAL_RUN = True
VIDEO_NAME = "100_NL_F_FA"
# --- END : !! TO MODIFY !! --- #


# --- Set the parameters --- #
DIMENSIONS = [108, 1820]
SCALE = 35
if REAL_RUN:
    TRIES = ""
else:
    TRIES = "/tries"


# --- Set the paths --- #
PATH_LABEL = [Path("../data/3_processed_positions{}/{}.csv".format(TRIES, VIDEO_NAME))]
STARTING_DATA_PATH = Path("../data/2_intermediate_top_down_lanes/lanes{}".format(TRIES))
STARTING_CALIBRATION_PATH = Path("../data/2_intermediate_top_down_lanes/calibration{}".format(TRIES))


try:
    # --- Generate and load the sets --- #
    DATA = generate_data(PATH_LABEL, STARTING_DATA_PATH, STARTING_CALIBRATION_PATH, take_all=False)
    SET = DataLoader(DATA, scale=SCALE, batch_size=1, dimensions=DIMENSIONS, standardization=False, augmentation=True, flip=True)
    print("The set is composed of {} images".format(len(DATA)))

    # --- Get every LANES --- #
    EVERY_LANES = [0] * len(DATA)
    for idx_sample in range(len(SET)):
        print(idx_sample)
        (lanes, labels) = SET[idx_sample]
        (lane, label) = (lanes[0].astype(np.uint8), labels[0].astype(int))
        # Modify the lane_magnifier if the head has been seen
        if label[0] >= 0:
            lane[:, label[1]] = [0, 0, 255]
        EVERY_LANES[idx_sample] = lane

    # --- Make the video --- #
    print("Making the video...")
    DESTINATION_VIDEO = Path("../data/5_model_output/videos/labelled_videos{}".format(TRIES))
    NAME_LABELLED_VIDEO = "labelled_augmented_flip{}.mp4".format(VIDEO_NAME)
    make_video(NAME_LABELLED_VIDEO, EVERY_LANES, destination=DESTINATION_VIDEO)

except FindPathDataError as find_path_data_error:
    print(find_path_data_error.__repr__())
except PaddingError as padding_error:
    print(padding_error.__repr__())
except AlreadyExistError as already_exist_error:
    print(already_exist_error.__repr__())
except FindPathError as find_path_error:
    print(find_path_error.__repr__())
