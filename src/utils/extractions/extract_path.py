from pathlib import Path
from os import listdir
import cv2


def get_frame_lane(image_name):
    image_name = image_name[1: -4]
    image_name = image_name.split("c")

    return image_name[0][: -1] + "," + image_name[1]


def extract_path(path_file_images):
    list_images = []
    list_images_name = []
    list_file = listdir(path_file_images)

    for file in list_file:
        if file[-4:] == ".jpg" or file[-4:] == ".JPG":
            path_image = path_file_images / file
            list_images.append(cv2.imread(str(path_image)))
            list_images_name.append(get_frame_lane(file))

    return list_images, list_images_name


if __name__ == "__main__":
    PATH_IMAGE = Path("../../../output/test/")
    (LIST_IMAGES, LIST_IMAGES_NAME) = extract_path(PATH_IMAGE)
    print("Number image", len(LIST_IMAGES))
    print(LIST_IMAGES_NAME)
