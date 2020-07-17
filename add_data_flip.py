"""
Add the flip label to a csv file.
"""
from pathlib import Path
import numpy as np

# Exceptions
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError

# The main file
from src.d3_processing_flip_images import add_swimming_way


# BEGIN : !! TO MODIFY !! #
# The name of the video
VIDEO_NAME = "vid1"

CHANGES = np.array([[1, 900], [2, 0], [2, 900]])
# END : !! TO MODIFY !! #


DESTINATION_CSV = Path("data/3_processed_positions")


try:
    add_swimming_way(VIDEO_NAME, DESTINATION_CSV, CHANGES)
except FindPathError as find_error:
    print(find_error.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())
