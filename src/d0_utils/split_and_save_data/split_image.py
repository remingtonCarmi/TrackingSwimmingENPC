"""
This module split an image into LANES and save the LANES.
"""
import cv2
from src.d0_utils.split_and_save_data.crop.crop_lines import crop


def split_and_save(image, margin, destination, frame, nb_lines):
    """
    Split the image into LANES and save the LANES

    Args:
        image (array): the image to split.

        margin (integer): the margin to take to be sure not to lose information.

        destination (WindowsPath): the path to the folder where the image will be stored.

        frame (integer): the number of the frame.

        nb_lines (integer): the number of lines in the pool.
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # List of the cuts, idx_lane start at 1 and not at 0 since there are any swimmers in this lane_magnifier.
    list_y = [int(1/nb_lines * idx_lane * image.shape[0]) for idx_lane in range(1, nb_lines)]

    # Crop the image
    list_images = crop(image, list_y, margin)

    # Save the cropped image except the first and the last one
    for idx_lane in range(1, nb_lines - 1):
        name = 'l%d' % idx_lane + '_f' + '0' * (4 - (len(str(frame)))) + str(frame) + '.jpg'
        cv2.imwrite(str(destination / name), list_images[idx_lane - 1])
