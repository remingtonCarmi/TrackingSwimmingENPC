from pathlib import Path
from os import listdir
import cv2
import numpy as np
from src.utils.extractions.exception_classes import FindErrorExtraction, EmptyFolder, NoMoreFrame


def register_image(image_name, frame, lane):
    # Verify that it is a jpg
    if image_name[-4:] == ".jpg" or image_name[-4:] == ".JPG":
        # Get the image information
        (image_frame, image_lane) = get_frame_lane(image_name)
        if frame < image_frame:
            return True
        elif frame == image_frame:
            return lane < image_lane
    return False


def get_frame_lane(image_name):
    image_name = image_name[1: -4]
    frame_lane = image_name.split("l")

    return [int(frame_lane[0][: -1]), int(frame_lane[1])]


def extract_path(path_file_images, frame, lane):
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
        if register_image(file, frame, lane):
            path_image = path_file_images / file
            list_images.append(cv2.imread(str(path_image)))
            list_images_name.append(get_frame_lane(file))

    if len(list_images_name) == 0:
        raise NoMoreFrame(path_file_images)

    return np.array(list_images), np.array(list_images_name)


if __name__ == "__main__":
    PATH_IMAGE = Path("../../../output/test/vid1/")

    try:
        (LIST_IMAGES, LIST_IMAGES_NAME) = extract_path(PATH_IMAGE, frame=21, lane=5)
        print(LIST_IMAGES_NAME)
        print("Number of images", len(LIST_IMAGES))
    except FindErrorExtraction as find_error:
        print(find_error.__repr__())
    except EmptyFolder as empty_folder:
        print(empty_folder.__repr__())
    except NoMoreFrame as no_more_frame:
        print(no_more_frame.__repr__())
