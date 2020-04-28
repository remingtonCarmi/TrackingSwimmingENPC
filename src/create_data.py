from src.utils.save_data.split_image import split_and_save
from src.utils.calibration_from_txt import calibrate_from_txt

import os
import cv2
from pathlib import Path


def create_data(path_video, path_txt, margin, nb_lines=10,
                time_begin=0, time_end=-1, destination=Path("../output/test/")):

    name_video = path_video.parts[-1][: -4]
    path_directory = Path(str(destination / name_video))

    # Create the directory where to save the data
    os.makedirs(path_directory)

    # Calibrate and extract images from the corrected video
    corrected_images = calibrate_from_txt(path_video, path_txt, time_begin, time_end)

    n = len(corrected_images)

    # Save the data
    for i in range(n):
        corrected_images[i] = cv2.cvtColor(corrected_images[i], cv2.COLOR_BGR2RGB)
        split_and_save(corrected_images[i], margin, path_directory, i, nb_lines)


if __name__ == "__main__":
    PATH_VIDEO = Path("../data/videos/vid0.mp4")
    PATH_TXT = Path("../data/calibration/vid0.txt")

    MARGIN = 0
    create_data(PATH_VIDEO, PATH_TXT, MARGIN, time_begin=11, time_end=12)
