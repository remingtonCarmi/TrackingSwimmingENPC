import numpy as np

from src.utils.extractions.extract_image import extract_image_video
from src.utils.perspective_correction.perspective_correction import get_top_down_image


def calibrate_from_txt(path_video, path_txt, time_begin=0, time_end=-1):

    file = open(path_txt, 'r')
    lines = file.readlines()
    homography = np.fromstring(lines[-2], dtype=float, sep=',')
    file.close()

    homography = np.reshape(homography, (3, 3))

    # Get the images
    print("Get the images ...")
    list_images = extract_image_video(path_video, time_begin, time_end)
    nb_images = len(list_images)

    # Transform the images
    print("Correction of images ...")
    for index_image in range(nb_images):
        list_images[index_image] = get_top_down_image(list_images[index_image], homography)

    return list_images
