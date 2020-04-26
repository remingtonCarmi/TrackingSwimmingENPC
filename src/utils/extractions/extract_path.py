from pathlib import Path
from os import listdir
import cv2


def extract_path(path_file_images):
    list_images = []
    list_file = listdir(path_file_images)

    for file in list_file:
        if file[-4:] == ".jpg" or file[-4:] == ".JPG":
            path_image = path_file_images / file
            list_images.append(cv2.imread(str(path_image)))

    return list_images


if __name__ == "__main__":
    PATH_IMAGE = Path("../../../output/test/")
    LIST_IMAGES = extract_path(PATH_IMAGE)
    print(len(LIST_IMAGES))
