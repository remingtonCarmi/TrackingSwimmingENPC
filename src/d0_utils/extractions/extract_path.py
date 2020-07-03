"""
This module takes the path of the images that have not been labelled.
"""
from pathlib import Path
from os import listdir
import numpy as np
from src.d0_utils.extractions.exceptions.exception_classes import FindPathExtractError, EmptyFolder, NoMoreFrame


def have_to_point_image(file_name, lane, frame):
    """
    Say if the file has to be pointed or not.

    Args:
        file_name (string): the name of the file.

        lane (integer): the index of the last lane pointed.

        frame (integer): the index of the last frame pointed.

    Returns:
        (boolean): say if the image has to be pointed.
    """
    # Verify that it is a jpg
    if file_name[-4:] == ".jpg" or file_name[-4:] == ".JPG":
        # Get the image information
        (image_lane, image_frame) = get_lane_frame(file_name)
        if lane < image_lane:
            return True
        elif lane == image_lane:
            return frame < image_frame
        else:
            return False
    return False


def get_lane_frame(image_name):
    """
    Get the indexes of the image

    Args:
        image_name (string): the name of the image

    Returns:
        (list of 2 integer): [index of the lane, index of the frame]
    """
    image_name = image_name[: -4]
    lane_frame = image_name.split("_")

    return [int(lane_frame[0][1:]), int(lane_frame[1][1:])]


def extract_path(path_file_images, lane, frame):
    """
    Extract the path of the images that have not been labelled.

    Args:
        path_file_images (WindowsPath): the path that leads to the images of lanes.

        lane (integer): the index of the last lane pointed.

        frame (integer): the index of the last frame pointed.

    Returns:
        (array): the list of the path that leads to the images that have not been pointed.

        (array): the list of the lanes and the frames of the images that have not been pointed.
    """
    if not path_file_images.exists():
        raise FindPathExtractError(path_file_images)

    # Initialize lists
    list_images_path = []
    list_lanes_frames = []
    list_file = listdir(path_file_images)

    # Verify that there is at least one image in the folder
    nb_files = len(list_file)
    idx_file = 0
    while idx_file < nb_files and not (list_file[idx_file][-4:] == ".jpg" or list_file[idx_file][-4:] == ".JPG"):
        idx_file += 1
    if idx_file == nb_files:
        raise EmptyFolder(path_file_images)

    # Get the path and the name of each image
    for file_name in list_file:
        if have_to_point_image(file_name, lane, frame):
            list_images_path.append(path_file_images / file_name)
            list_lanes_frames.append(get_lane_frame(file_name))

    if len(list_images_path) == 0:
        raise NoMoreFrame(path_file_images)

    return np.array(list_images_path), np.array(list_lanes_frames)


if __name__ == "__main__":
    PATH_IMAGE = Path("../../../data/1_intermediate_top_down_lanes/lanes/tries/vid0")

    try:
        (LIST_IMAGES_PATH, LIST_LANES_FRAMES) = extract_path(PATH_IMAGE, 1, 525)
        print(LIST_IMAGES_PATH)
        print(LIST_LANES_FRAMES)
        print("Number of images", len(LIST_IMAGES_PATH))
    except FindPathExtractError as find_error:
        print(find_error.__repr__())
    except EmptyFolder as empty_folder:
        print(empty_folder.__repr__())
    except NoMoreFrame as no_more_frame:
        print(no_more_frame.__repr__())
