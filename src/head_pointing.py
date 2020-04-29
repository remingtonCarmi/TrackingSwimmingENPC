"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import numpy as np
from src.utils.extractions.exception_classes import FindError, EmptyFolder
from src.utils.point_selection.head_selection import head_selection
from src.utils.store_load_matrix.exception_classes import AlreadyExistError
from src.utils.extractions.extract_path import extract_path
from src.utils.point_selection.instructions.instructions import instructions_head
from src.utils.store_load_matrix.fill_csv import fill_csv, last_line


def heads_selection(nb_images_to_point, images):
    # Instructions
    instructions_head()

    # Head selection
    list_head = [0] * nb_images_to_point
    print("Head selection ...")
    for index_image in range(nb_images_to_point):
        list_head[index_image] = head_selection(images[index_image])

    return np.array(list_head)


def head_pointing(path_images, nb_images=-1, destination_csv=Path("../output/test/")):

    # Check that the folder exists
    if not destination_csv.exists():
        raise FindError(destination_csv)

    # Create the csv if it does not exist
    name_csv = path_images.parts[-1] + ".csv"
    destination_csv_file = destination_csv / name_csv
    if not destination_csv_file.exists():
        fill_csv(name_csv, [], [], destination_csv)

    # Get the last line that was registered
    (last_frame, last_lane) = last_line(destination_csv_file)

    # Get the images
    print("Get the images ...")
    (list_images, list_images_name) = extract_path(path_images, last_frame, last_lane)
    nb_total_images = len(list_images)
    if nb_images == -1:
        nb_images = float('inf')
    nb_pointed_image = min(nb_images, nb_total_images)

    # Select the heads
    heads = heads_selection(nb_pointed_image, list_images)

    return heads


if __name__ == "__main__":
    PATH_IMAGES = Path("../output/images/vid0/")

    try:
        LIST_HEAD = head_pointing(PATH_IMAGES, 3)
        print(LIST_HEAD)
    except FindError as find_error:
        print(find_error.__repr__())
    except EmptyFolder as empty_folder:
        print(empty_folder.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
