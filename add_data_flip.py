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
VIDEO_NAME = "100NL_FAF"

CHANGES = np.array([[1, 0], [1, 750], [2, 0], [2, 750], [3, 0], [3, 750], [4, 0], [4, 750], [5, 0], [5, 750], [6, 0], [6, 750], [7, 0], [7, 750], [8, 0], [8, 750]])
# END : !! TO MODIFY !! #


DESTINATION_CSV = Path("data/3_processed_positions")


try:
    add_swimming_way(VIDEO_NAME, DESTINATION_CSV, CHANGES)
except FindPathError as find_error:
    print(find_error.__repr__())
except AlreadyExistError as exist_error:
    print(exist_error.__repr__())
