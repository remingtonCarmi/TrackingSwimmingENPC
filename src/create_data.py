from src.utils.save_data.split_image import split_and_save
from src.utils.calibration_from_txt import calibrate_from_txt

from src.utils.save_data.exception_classes import FileAlreadyExists
from src.utils.extractions.exceptions.exception_classes import TimeError
from src.utils.extractions.exceptions.exception_classes import FindErrorExtraction


import os
import cv2
from pathlib import Path


def create_data(path_video, path_txt, margin, nb_lines=10,
                time_begin=0, time_end=-1, destination=Path("../output/test/")):

    name_video = path_video.parts[-1][: -4]
    path_directory = Path(str(destination / name_video))

    # Check if the data has already been generated

    if os.path.exists(path_directory):
        first_frame = time_begin * 25
        first_frame_name = 'l1_f' + '0' * (4 - (len(str(first_frame)))) + str(first_frame) + '.jpg'
        first_frame_path = Path(str(path_directory / first_frame_name))
        if os.path.exists(first_frame_path):
            raise FileAlreadyExists(first_frame_path)
        else:
            print("Caution! You already have other saved lines in the folder. "
                  "\nThe new ones are added with the previous ones.")
    else:
        # Create the directory where to save the data
        os.makedirs(path_directory)

    # Calibrate and extract images from the corrected video
    corrected_images = calibrate_from_txt(path_video, path_txt, time_begin, time_end)

    n = len(corrected_images)

    # Save the data
    for i in range(n):
        corrected_images[i] = cv2.cvtColor(corrected_images[i], cv2.COLOR_BGR2RGB)
        split_and_save(corrected_images[i], margin, path_directory, time_begin * 25 + i, nb_lines)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid0.mp4")
    PATH_TXT = Path("../data/calibration/vid0.txt")

    MARGIN = 0

    try:
        create_data(PATH_VIDEO, PATH_TXT, MARGIN, time_begin=2, time_end=3)

    except FileAlreadyExists as already_exists:
        print(already_exists.__repr__())

    except TimeError as time_error:
        print(time_error.__repr__())

    except FindErrorExtraction as find_error:
        print(find_error.__repr__())
