from pathlib import Path
from os import listdir
import cv2
import numpy as np
from src.utils.extractions.exception_classes import FindErrorExtraction, EmptyFolder, NoMoreFrame


def register_image(image_name, lane, frame):
    # Verify that it is a jpg
    if image_name[-4:] == ".jpg" or image_name[-4:] == ".JPG":
        # Get the image information
        (image_lane, image_frame) = get_frame_lane(image_name)
        if lane < image_lane:
            return True
        elif lane == image_lane:
            return frame < image_frame
    return False


def get_frame_lane(image_name):
    image_name = image_name[: -4]
    lane_frame = image_name.split("_")
    return [int(lane_frame[0][1:]), int(lane_frame[1][1:])]


def extract_path(path_file_images, lane, frame):
    if not path_file_images.exists():
        raise FindErrorExtraction(path_file_images)

    # Initialize lists
    list_images = []
    list_images_name = []
    list_file = listdir(path_file_images)

    # Verify that the file is not empty
    nb_files = len(list_file)
    idx_file = 0
    while idx_file < nb_files and not (list_file[idx_file][-4:] == ".jpg" or list_file[idx_file][-4:] == ".JPG"):
        idx_file += 1
    if idx_file == nb_files:
        raise EmptyFolder(path_file_images)

    for file in list_file:
        if register_image(file, lane, frame):
            path_image = path_file_images / file
            list_images.append(cv2.imread(str(path_image)))
            list_images_name.append(get_frame_lane(file))

    if len(list_images_name) == 0:
        raise NoMoreFrame(path_file_images)

    return np.array(list_images), np.array(list_images_name)


if __name__ == "__main__":
    PATH_IMAGE = Path("../../../output/test/vid1/")

    try:
        (LIST_IMAGES, LIST_IMAGES_NAME) = extract_path(PATH_IMAGE, lane=4, frame=5)
        print(LIST_IMAGES_NAME)
        print("Number of images", len(LIST_IMAGES))
    except FindErrorExtraction as find_error:
        print(find_error.__repr__())
    except EmptyFolder as empty_folder:
        print(empty_folder.__repr__())
    except NoMoreFrame as no_more_frame:
        print(no_more_frame.__repr__())
