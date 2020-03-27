""" main for red boxes"""

import os
import shutil
from src.extract_image import extract_image_video
from src.red_boxes.crop_lines import crop_list


def clean_test(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


if __name__ == "__main__":
    FOLDER = "..\\..\\test\\red_boxes\\"
    FRAME_NAME = FOLDER + "frame2.jpg"
    clean_test(FOLDER)

    LIST_IMAGES = extract_image_video("..\\data\\videos\\vid0", 0, 0.2, True, FOLDER)
    LIST_IMAGES_CROP = crop_list(LIST_IMAGES)

