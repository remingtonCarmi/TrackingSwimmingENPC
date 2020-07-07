"""
This code allows the user to point at the head of the swimmers.
"""
from pathlib import Path
import numpy as np
import cv2

# Exceptions
from src.d0_utils.extractions.exceptions.exception_classes import EmptyFolder, NoMoreFrame
from src.d0_utils.store_load_data.exceptions.exception_classes import FindPathError, AlreadyExistError, NothingToAddError

# To extract the path of the images
from src.d0_utils.extractions.extract_path import extract_path, get_lane_frame

# To point at the head of the swimmers
from src.d0_utils.point_selection.instructions.instructions import instructions_head
from src.d0_utils.point_selection.head_selection import head_selection

# To save the pointing
from src.d0_utils.store_load_data.fill_csv import create_csv, last_line, fill_csv


def heads_selection(images_paths):
    """
    Allow the user to select the head on the images.

    Args:
        images_paths (list of WindowsPath): list of the paths that lead to the images.

    Returns:
        (array): the list of point that where pointed.
    """
    nb_images = len(images_paths)
    # Instructions
    instructions_head()

    # Head selection
    list_head = []
    index_image = 0
    stop = False

    # Continue until there is no more image or until the user wants to stop
    while index_image < nb_images and not stop:
        # Load the image
        image = cv2.imread(str(images_paths[index_image]))

        # Get the name of the image
        image_name = images_paths[index_image].parts[-1]
        (lane, frame) = get_lane_frame(image_name)

        # Plot the image and select the head
        (point_head, stop) = head_selection(image, lane, frame)

        # Register if the user wants to continue
        if not stop:
            list_head.append(point_head[0])

        # Update the image index
        index_image += 1

    return np.array(list_head)


def head_pointing(path_images, destination_csv=None):
    """
    Allows the user to point at the head of the swimmers and save the data.

    Args:
        path_images (WindowsPath): the path that leads to the images.

        destination_csv (WindowsPath): the path where the csv file is/will be  registered.

    Returns:
        list_points (array): list of the points that have been pointed.
    """
    if destination_csv is None:
        destination_csv = Path("../data/2_processed_positions/tries")

    # Check that the folder exists
    if not path_images.exists():
        raise FindPathError(path_images)

    if not destination_csv.exists():
        raise FindPathError(destination_csv)

    # Create the csv if it does not exist
    name_csv = path_images.parts[-1] + ".csv"
    csv_path = destination_csv / name_csv
    if not csv_path.exists():
        create_csv(name_csv, destination_csv)

    # Get the last line that was registered
    (last_lane, last_frame) = last_line(csv_path)

    # Get the images
    (list_images_path, list_lanes_frames) = extract_path(path_images, last_lane, last_frame)

    # Select the heads
    print("Head selection ...")
    list_points = heads_selection(list_images_path)
    nb_pointed_image = len(list_points)

    # Register in the csv
    fill_csv(csv_path, list_lanes_frames[: nb_pointed_image], list_points)

    return list_points


if __name__ == "__main__":
    PATH_IMAGES = Path("../data/1_intermediate_top_down_lanes/lanes/tries/vid1")

    try:
        LIST_HEAD = head_pointing(PATH_IMAGES)
        print(LIST_HEAD)
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
