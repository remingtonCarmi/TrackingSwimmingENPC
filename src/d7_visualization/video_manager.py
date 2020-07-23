"""
This module contains the class VideoManager
"""
import numpy as np


class VideoManager:
    """
    Class to manage the creation of a video with the predictions.
    """
    def __init__(self, nb_images):
        """
        Construct the number of images that the video will contain.
        """
        # Will contain all the lanes with the predictions
        self.lanes_with_preds = [0] * nb_images

    def update(self, idx_image, lane, index_preds, index_regression_pred):
        """
        Update the list of lanes.

        Args:
            idx_image (integer): the index of the image.

            lane (array): the image.

            index_preds (array): the index of the columns where the head should be.

            index_regression_pred (integer): the column where the head should be.
        """
        # -- For the classification problem -- #
        # Modify the color of the lane
        lane[:, index_preds] += 40
        np.clip(lane, 0, 255, lane)

        # Add the main list
        self.lanes_with_preds[idx_image] = lane.astype(np.uint8)

        # -- For the regression problem -- #
        self.lanes_with_preds[idx_image][:, index_regression_pred] = [0, 0, 255]
