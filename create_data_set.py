"""
This script allows the user to create the data set.

That is to say, the user can create:
    - a txt file that registers the information about the calibration
    - a folder full of the LANES of the video
    - a csv file that registers the information about the position of the swimmers
"""
from pathlib import Path

# Exceptions
from src.d0_utils.extractions.exceptions.exception_classes import FindPathExtractError, TimeError, EmptyFolder, NoMoreFrame
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError, NothingToAddError

# To calibrate the video
from src.d2_intermediate_calibration import calibrate_video_text

# To create the image LANES
from src.d2_intermediate_lanes import create_data

# To point at the heads of the swimmers
from src.d3_processing_head_pointing import head_pointing


# BEGIN : !! TO MODIFY !! #
# The name of the video
NAME_VIDEO = "S5580002"

# Time range for pointing
POINTING_STARTING_TIME = 103
POINTING_ENDING_TIME = 110
# END : !! TO MODIFY !! #


# --- Create the txt file --- #
PATH_VIDEO = Path("data/1_raw_videos/{}.mp4".format(NAME_VIDEO))
DESTINATION_TXT = Path("data/2_intermediate_top_down_lanes/calibration")

try:
    print(" --- Create the txt file for calibration --- ")
    calibrate_video_text(PATH_VIDEO, 2, DESTINATION_TXT)  # The calibration is done on an image at second 1.
except FindPathExtractError as video_find_extract_error:
    print(video_find_extract_error.__repr__())
except TimeError as time_error:
    print(time_error.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())


# --- Create the LANES image --- #
PATH_TXT = Path("data/2_intermediate_top_down_lanes/calibration/{}.txt".format(NAME_VIDEO))
DESTINATION_LANES = Path("data/2_intermediate_top_down_lanes/LANES")

try:
    print(" --- Save the LANES as jpg files --- ")
    MARGIN = 0

    create_data(PATH_VIDEO, PATH_TXT, MARGIN, destination=DESTINATION_LANES,
                time_begin=POINTING_STARTING_TIME,
                time_end=POINTING_ENDING_TIME)
except FindPathError as find_error:
    print(find_error.__repr__())
except AlreadyExistError as already_exists:
    print(already_exists.__repr__())
except TimeError as time_error:
    print(time_error.__repr__())


# --- Create the csv file --- #
PATH_LANES = DESTINATION_LANES / NAME_VIDEO
DESTINATION_CSV = Path("data/3_processed_positions")

try:
    print(" --- Create the csv file for pointing --- ")
    LIST_HEAD = head_pointing(PATH_LANES, destination_csv=DESTINATION_CSV)
except FindPathError as find_error:
    print(find_error.__repr__())
except EmptyFolder as empty_folder:
    print(empty_folder.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())
except NothingToAddError as nothing_to_add:
    print(nothing_to_add.__repr__())
except NoMoreFrame as no_more_frame:
    print(no_more_frame.__repr__())
