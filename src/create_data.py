from src.utils.save_data.split_image import split_and_save
from src.utils.extract_image import extract_image_video
from src.calibration import calibrate_video

import os
import cv2
from pathlib import Path


def create_data(path_video, margin, nb_lines=10,
                is_corrected=False,
                time_begin=0, time_end=-1, destination=Path("../output/images/")):

    name_video = path_video.parts[-1][: -4]
    path_directory = Path(str(destination / name_video))

    # Create the directory where to save the data
    os.makedirs(path_directory)

    # Extract images from the video
    if not is_corrected:
        corrected_images = calibrate_video(path_video, time_begin, time_end, create_video=False, creat_txt=True)
    else:
        corrected_images = extract_image_video(path_video, time_begin, time_end)

    n = len(corrected_images)

    # Save the data
    for i in range(n):
        corrected_images[i] = cv2.cvtColor(corrected_images[i], cv2.COLOR_BGR2RGB)
        split_and_save(corrected_images[i], margin, path_directory, i, nb_lines)


if __name__ == "__main__":
    PATH_VIDEO = Path("../../data/videos/vid0.mp4")
    # PATH_VIDEO_CALIBRATED = Path("../output/videos/corrected_vid0.mp4")

    MARGIN = -10
    create_data(PATH_VIDEO, MARGIN, is_corrected=False, time_begin=11, time_end=13)
