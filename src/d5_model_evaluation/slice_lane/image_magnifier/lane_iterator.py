"""
This module contains the class LaneIterator.
"""


class LaneIterator:
    """
    Class that returns the limits of the slice of a lane.
    """
    def __init__(self, nb_sub_images, window_size, recovery, image_horiz_size):
        """
        Construct the lane iterator.

        Args:
            nb_sub_images (integer): the number of sub-images that has been sliced.

            window_size (integer): the width of the sub-image.

            recovery (integer): the number of pixels to be taken twice per sub-image.

            image_horiz_size (integer): the horizontal size of the original image.
        """
        self.nb_sub_images = nb_sub_images
        self.window_size = window_size
        self.recovery = recovery
        self.image_horiz_size = image_horiz_size

    def get_limits(self, idx):
        """
        Compute the limits of specific a sub-image.

        Args:
            idx (integer): the specific sub-image.

        Returns:
             (integer): the beginning limit.

             (integer): the ending limit.
        """
        if idx < self.nb_sub_images - 1:
            pixel_step = self.window_size - self.recovery

            return idx * pixel_step, idx * pixel_step + self.window_size
        elif idx == self.nb_sub_images - 1:
            return - self.window_size, None
        else:
            return None, None


if __name__ == "__main__":
    # Imports
    from pathlib import Path
    import numpy as np
    import cv2

    # Data
    PATH_IMAGE = Path("../../../../data/2_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0297.jpg")
    PATH_IMAGE = Path("../../../../data/5_model_output/tries/transformed_images/transformed_l1_f0275.jpg")

    LANE = cv2.imread(str(PATH_IMAGE))
    LABEL = np.array([54, 1570])
    LABEL = np.array([49, 648])
    NB_SUB_IMAGES = 10
    WINDOW_SIZE = 200
    RECOVERY = 100

    LANE_ITERATOR = LaneIterator(NB_SUB_IMAGES, WINDOW_SIZE, RECOVERY, 1820)

    for idx_image in range(LANE_ITERATOR.nb_sub_images):
        print("Limits", LANE_ITERATOR.get_limits(idx_image))

