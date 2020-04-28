"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import numpy as np
from src.utils.extractions.extract_image import TimeError
from src.utils.extractions.exception_classes import FindError, EmptyFolder
from src.utils.point_selection.head_selection import head_selection
from src.utils.store_load_matrix.exception_classes import AlreadyExistError
from src.utils.extractions.extract_path import extract_path
from src.utils.point_selection.instructions.instructions import instructions_head
from src.utils.store_load_matrix.fill_csv import fill_csv


def head_pointing(path_images, nb_images=-1, destination_csv=Path("../output/test/"), create_csv=False):
    if create_csv:
        # Check that the folder exists
        if not destination_csv.exists():
            raise FindError(destination_csv)

        # Verify is the csv does not exist
        name_csv = path_images.parts[-1] + ".csv"
        destination_csv_file = destination_csv / name_csv
        if destination_csv_file.exists():
            raise AlreadyExistError(destination_csv_file)

    # Get the images
    print("Get the images ...")
    (list_images, list_images_name) = extract_path(path_images)
    nb_total_images = len(list_images)
    if nb_images == -1:
        nb_images = float('inf')
    nb_pointed_image = min(nb_images, nb_total_images)

    # Instructions
    instructions_head()

    # Head selection
    list_head = [0] * nb_pointed_image
    print("Head selection ...")
    for index_image in range(nb_pointed_image):
        list_head[index_image] = head_selection(list_images[index_image])
    list_head = np.array(list_head)

    if create_csv:
        fill_csv(name_csv, list_head, list_images_name, destination_csv)

    return list_head


if __name__ == "__main__":
    PATH_IMAGES = Path("../output/images/vid0/")

    try:
        LIST_HEAD = head_pointing(PATH_IMAGES, 3, create_csv=True)
        print(LIST_HEAD)
    except FindError as find_error:
        print(find_error.__repr__())
    except EmptyFolder as empty_folder:
        print(empty_folder.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
