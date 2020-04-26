"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import numpy as np
from src.utils.extractions.extract_image import TimeError
from src.utils.extractions.exception_classes import VideoFindError
from src.utils.point_selection.head_selection import head_selection
from src.utils.store_load_matrix.fill_txt import TXTExistError
from src.utils.extractions.extract_path import extract_path
from src.utils.point_selection.instructions.instructions import instructions_head


def head_pointing(path_images, nb_images=-1, destination_csv=Path("../output/test/"), create_csv=False):
    # Instruction
    instructions_head()

    # Get the images
    print("Get the images ...")
    list_images = extract_path(path_images)
    nb_total_images = len(list_images)
    if nb_images == -1:
        nb_images = float('inf')
    nb_pointed_image = min(nb_images, nb_total_images)

    # Head selection
    list_head = [0] * nb_images
    print("Head selection ...")
    for index_image in range(nb_pointed_image):
        list_head[index_image] = head_selection(list_images[index_image])
    list_head = np.array(list_head)

    if create_csv:
        name_csv = path_images.parts[-1]
        path_csv = destination_csv / name_csv
        # store_calibration_csv(path_csv, list_head)

    return list_head


if __name__ == "__main__":
    PATH_IMAGES = Path("../output/test/")
    DESTINATION_CSV = Path("../data/videos/corrected/")
    try:
        LIST_HEAD = head_pointing(PATH_IMAGES, 3)
        print(LIST_HEAD)
    except TimeError as time_error:
        print(time_error.__repr__())
    except VideoFindError as video_find_error:
        print(video_find_error.__repr__())
    except TXTExistError as exist_error:
        print(exist_error.__repr__())
