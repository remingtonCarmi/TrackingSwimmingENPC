"""
This module slice a lane with its label into smaller sub-images.
"""
from pathlib import Path
import numpy as np
import cv2


class ImageMagnifier:
    """
    The class that enables to slice a lane with its label.
    """
    def __init__(self, lane, label, window_size, recovery):
        """
        Register the image.

        Args:
            lane (array): the input lane.

            label (list: (integer, integer)): the position of the head : [y_head, x_head]

            window_size (integer): the width of the sub-image.

            recovery (integer): the number of pixels to be taken twice per sub-image.
        """
        # Get the lane
        self.lane = lane
        self.dimensions = lane.shape[: 2]

        # Register the label
        self.label = label

        # Parameters of the magnifier
        self.window_size = window_size
        self.recovery = recovery

        # Get the number of sub-image
        self.nb_sub_images = len(self)

    def __len__(self):
        """
        Computes the number of sub-image for the lane.

        Returns:
            (integer): the number of sub-image.
        """
        nb_stake = int(np.ceil(self.dimensions[1] / (self.window_size - self.recovery)))
        return nb_stake - self.window_size // (self.window_size - self.recovery) + 1

    def __getitem__(self, idx):
        """
        Give the sub-image and its label with a specific index.

        Args:
            idx (integer): the index wanted.

        Returns:
            (array): the sub-image.

            (list: (integer, integer, integer)): (is_in_image, is_not_in_image, column)
                column is the index of the column of pixel is the head is located.
                If present is False, column = -1
        """
        if idx < self.nb_sub_images - 1:
            pixel_step = self.window_size - self.recovery

            # Compute the label
            if idx * pixel_step < self.label[1] < idx * pixel_step + self.window_size:
                label = [1, 0, self.label[1] - idx * pixel_step]
            else:
                label = [0, 1, -1]
            return self.lane[:, idx * pixel_step: idx * pixel_step + self.window_size], label
        elif idx == self.nb_sub_images - 1:
            # Compute the label
            if self.dimensions[1] - self.window_size < self.label[1] < self.dimensions[1]:
                label = [1, 0, self.label[1] - self.dimensions[1] - self.window_size]
            else:
                label = [0, 1, -1]
            return self.lane[:, - self.window_size:], label
        else:
            raise StopIteration


if __name__ == "__main__":
    # Data
    PATH_IMAGE = Path("../../../../../data/1_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0297.jpg")
    # PATH_IMAGE = Path("../../../../data/4_model_output/tries/scaled_images/scaled_l1_f0275.jpg")

    LANE = cv2.imread(str(PATH_IMAGE))
    LABEL = np.array([54, 1570])
    # LABEL = np.array([49, 648])
    WINDOW_SIZE = 200
    RECOVERY = 100

    IMAGE_MAGNIFIER = ImageMagnifier(LANE, LABEL, WINDOW_SIZE, RECOVERY)

    print("Nb sub image", len(IMAGE_MAGNIFIER))

    for (idx_image, (sub_lane, sub_label)) in enumerate(IMAGE_MAGNIFIER):
        print("Image n° {}. Present = {}".format(idx_image, sub_label[0]))
        print("Dimensions", sub_lane.shape)
        if sub_label[0]:
            sub_lane[:, sub_label[2]] = [0, 0, 255]
        cv2.imshow("Image n' {}. Present = {}".format(idx_image, sub_label[0]), sub_lane)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
