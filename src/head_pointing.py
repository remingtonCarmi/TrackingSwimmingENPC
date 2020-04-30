"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
from pathlib import Path
import numpy as np
from src.utils.extractions.exception_classes import EmptyFolder, NoMoreFrame
from src.utils.point_selection.head_selection import head_selection
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, NothingToAdd, FindErrorStore
from src.utils.extractions.extract_path import extract_path
from src.utils.point_selection.instructions.instructions import instructions_head
from src.utils.store_load_matrix.fill_csv import create_csv, last_line, fill_csv


def heads_selection(images):
    nb_images = len(images)
    # Instructions
    instructions_head()

    # Head selection
    list_head = []
    index_image = 0
    stop = False
    # To avoid having a -2, -2 in the last line
    while index_image < nb_images and not stop:
        (point_head, stop) = head_selection(images[index_image])
        if not stop:
            list_head.append(point_head[0])
        index_image += 1

    return np.array(list_head)


def head_pointing(path_images, destination_csv=Path("../output/test/")):
    # Check that the folder exists
    if not destination_csv.exists():
        raise FindErrorStore(destination_csv)

    # Create the csv if it does not exist
    name_csv = path_images.parts[-1] + ".csv"
    csv_path = destination_csv / name_csv
    if not csv_path.exists():
        create_csv(name_csv, destination_csv)

    # Get the last line that was registered
    (last_frame, last_lane) = last_line(csv_path)

    # Get the images
    print("Load the images ...")
    (list_images, list_images_name) = extract_path(path_images, last_frame, last_lane)

    # Select the heads
    print("Head selection ...")
    list_points = heads_selection(list_images)
    nb_pointed_image = len(list_points)

    # Register in the csv
    print("Write the newly pointed heads ...")
    fill_csv(csv_path, list_images_name[: nb_pointed_image], list_points)

    return list_points


if __name__ == "__main__":
    PATH_IMAGES = Path("../output/test/vid1/")

    try:
        LIST_HEAD = head_pointing(PATH_IMAGES)
        print(LIST_HEAD)
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
