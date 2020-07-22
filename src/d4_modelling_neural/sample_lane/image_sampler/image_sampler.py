"""
This module samples a lane with its label into smaller sub-images.
"""
from pathlib import Path
import numpy as np
import cv2
import numpy.random as rd


class ImageSampler:
    """
    The class that enables to sample a lane with its label.
    """
    def __init__(self, lane, label, window_size, nb_samples, distribution, margin, close_to_head):
        """
        Register the image.

        Args:
            lane (array): the input lane_magnifier.

            label (list: (integer, integer)): the position of the head : [y_head, x_head]

            window_size (integer): the width of the sub-image.

            nb_samples (integer): the number of sub_images to be created.

            distribution (float in [0, 1]): the percentage of head returned.

            margin (integer): the margin to be sure that the head is not in the border of the image.

            close_to_head (boolean): if True, the column with be chosen near the head.
        """
        # Get the lane_magnifier
        self.lane = lane
        self.dimensions = lane.shape[: 2]

        # Register the label
        self.label = label

        # Parameters of the sampler
        self.window_size = window_size
        self.nb_samples = nb_samples
        self.distribution = distribution
        self.margin = margin

        # Parameter to choose the column
        self.right_column = self.lane.shape[1] - self.window_size
        self.left_column = 0

        if close_to_head:
            self.left_column = max(0, self.label[1] - 2 * window_size)
            self.right_column = min(self.right_column, self.label[1] + window_size)

    def __len__(self):
        """
        Computes the number of sub-image for the lane_sampler.

        Returns:
            (integer): the number of sub-image.
        """
        return self.nb_samples

    def __getitem__(self, idx):
        """
        Give the sub-image and its label at random.

        Args:
            idx (integer): an index.

        Returns:
            (array): the sub-image.

            (list: (integer, integer, integer)): (is_in_image, is_not_in_image, column)
                column is the index of the column of pixel is the head is located.
                If present is False, column = -1
        """
        if idx < self.nb_samples:
            # Select a head
            if rd.random() <= self.distribution:
                # Select a random column
                column = self.get_column_head()

                # Create the label
                label = [1, 0, self.label[1] - column]

            # Select a sub-image with no head
            else:
                # Select a random column
                column = self.get_column_no_head()

                # Create the label
                label = [0, 1, - 1]
            return self.lane[:, column: column + self.window_size], label
        else:
            raise StopIteration

    def get_column_head(self):
        """
        Take a random column with which the window will contain a head.

        Returns:
            column (integer): the column.
        """
        column = rd.randint(self.left_column, self.right_column)

        # We continue until the window contains the head with a margin
        while (not self.contain_head(column)) or self.is_black(column):
            column = rd.randint(self.left_column, self.right_column)

        return column

    def get_column_no_head(self):
        """
        Take a random column with which the window will not contain a head.

        Returns:
            column (integer): the column.
        """
        column = rd.randint(self.left_column, self.right_column)

        # We continue until the window is at the left or at the right of the head with a margin
        while (not self.is_left(column) and not self.is_right(column)) or self.is_black(column):
            column = rd.randint(self.left_column, self.right_column)

        return column

    def contain_head(self, column):
        """
        Says if the window starting with the column contains a head.
        """
        return column + self.margin <= self.label[1] <= column + self.window_size - self.margin

    def is_left(self, column):
        """
        Says if the window starting with the column is at the left of the head.
        """
        return column + self.window_size + self.margin <= self.label[1]

    def is_right(self, column):
        """
        Says if the window starting with the column is at the right of the head.
        """
        return self.label[1] <= column - self.margin

    def is_black(self, column):
        """
        Says if the window starting with the column is black.
        """
        return np.sum(self.lane[:, column: column + self.window_size]) == 0


if __name__ == "__main__":
    # Data
    PATH_IMAGE1 = Path("../../../../../data/5_model_output/tries/transformed_images/transformed_l1_f0275.jpg")
    PATH_IMAGE2 = Path("../../../../../data/5_model_output/tries/transformed_images/transformed_l1_f0107.jpg")
    PATH_IMAGE3 = Path("../../../../../data/5_model_output/tries/transformed_images/transformed_l8_f1054.jpg")
    PATH_IMAGE4 = Path("../../../../../data/5_model_output/tries/transformed_images/transformed_l1_f0339.jpg")

    LANE = cv2.imread(str(PATH_IMAGE4))

    LABEL = np.array([49, 648])
    LABEL = np.array([49, 768])
    LABEL = np.array([53, 950])
    LABEL = np.array([41, 1163])

    WINDOW_SIZE = 30
    NB_SAMPLE = 10
    DISTRIBUTION = 1
    MARGIN = 5
    CLOSE_TO_HEAD = True

    IMAGE_SAMPLER = ImageSampler(LANE, LABEL, WINDOW_SIZE, NB_SAMPLE, DISTRIBUTION, MARGIN, CLOSE_TO_HEAD)

    print("Nb sub image", len(IMAGE_SAMPLER))

    for (idx_image, (sub_lane, sub_label)) in enumerate(IMAGE_SAMPLER):
        print("Image n° {}. Present = {}".format(idx_image, sub_label[0]))
        print("Dimensions", sub_lane.shape)
        if sub_label[0]:
            sub_lane[:, sub_label[2]] = [0, 0, 255]
        cv2.imshow("Image n' {}. Present = {}".format(idx_image, sub_label[0]), sub_lane)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
