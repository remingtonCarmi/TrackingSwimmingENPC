"""
This script allows the user to create the data set.

That is to say, the user can create:
    - a txt file that registers the information about the calibration
    - a folder full of the lanes of the video
    - a csv file that registers the information about the position of the swimmers
"""
from pathlib import Path
from src.calibration import calibrate_video
from src.create_data import create_data
from src.head_pointing import head_pointing
from src.utils.extractions.exception_classes import TimeError, EmptyFolder, NoMoreFrame, FindErrorExtraction
from src.utils.save_data.exception_classes import FileAlreadyExists
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, NothingToAdd, FindErrorStore


# The name of the video
NAME_VIDEO = "vid1"


# --- Create the txt file --- #
PATH_VIDEO = Path("data/videos/{}.mp4".format(NAME_VIDEO))
DESTINATION_TXT = Path("data/calibration/")
try:
    print(" --- Create the txt file for calibration --- ")
    calibrate_video(PATH_VIDEO, destination_txt=DESTINATION_TXT, create_txt=True, time_begin=3, time_end=4)
except FindErrorExtraction as video_find_error:
    print(video_find_error.__repr__())
except TimeError as time_error:
    print(time_error.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())


# --- Create the lanes images --- #
PATH_TXT = Path("data/calibration/{}.txt".format(NAME_VIDEO))
DESTINATION_LANES = Path("data/lanes/")
try:
    print(" --- Save the lanes as jpg files --- ")
    MARGIN = 0

    create_data(PATH_VIDEO, PATH_TXT, MARGIN, destination=DESTINATION_LANES, time_begin=5, time_end=6)
except FileAlreadyExists as already_exists:
    print(already_exists.__repr__())
except TimeError as time_error:
    print(time_error.__repr__())
except FindErrorStore as find_error:
    print(find_error.__repr__())


# --- Create the csv file --- #
PATH_LANES = DESTINATION_LANES / NAME_VIDEO
DESTINATION_CSV = Path("data/head_points")
try:
    print(" --- Create the csv file for pointing --- ")
    LIST_HEAD = head_pointing(PATH_LANES, destination_csv=DESTINATION_CSV)
except FindErrorStore as find_error:
    print(find_error.__repr__())
except EmptyFolder as empty_folder:
    print(empty_folder.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())
except NothingToAdd as nothing_to_add:
    print(nothing_to_add.__repr__())
except NoMoreFrame as no_more_frame:
    print(no_more_frame.__repr__())
