"""
This module contains the GraphicManager class.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# To get the left limit and the video length
from src.d7_visualization.tools.get_meters_video import get_meters_video


class GraphicManager:
    """
    Class to manage the graphic data.
    """
    def __init__(self, prediction_memories):
        """
        Construct the parameters and the data for the figure.

        Args:
            prediction_memories (PredictionMemories): THE ATTRIBUTE DATA HAS TO BE FILLED i.e
                THE VIDEO HAS TO BE MADE BEFORE.
        """
        # Parameters
        self.fps = prediction_memories.fps
        (self.left_limit, length_video) = get_meters_video(prediction_memories.calibration_path)
        self.scale = prediction_memories.scale
        self.added_pad = prediction_memories.added_pad

        # Data for the figure
        self.begin_frame = prediction_memories.begin_frame
        self.nb_images = len(prediction_memories.preds)

        self.pos_predictions = np.zeros((self.nb_images, 2))
        self.pos_regression = np.zeros(self.nb_images)
        self.pos_real = np.zeros(self.nb_images)
        self.time = np.zeros(self.nb_images)

        # Label in the original coordinates
        labels = prediction_memories.data[:, 2]
        unlabelled_lanes = np.where(labels < 0)
        # Label in the transformed coordinates
        labels /= prediction_memories.unscaled_factor
        labels += prediction_memories.added_pad
        # Put the unlabelled lanes down to zero
        labels[unlabelled_lanes] = 0

        self.fill_positions(prediction_memories.preds, labels)

    def fill_positions(self, predictions, labels):
        """
        Fill the data for the graphic.

        Args:
            predictions (list of list of integers): the list of predictions by frame,
                one element is [lane, index_classification_left, index_classification_right, index_regression].

            labels (array of integers): the list of the head positions.
        """
        # Count the index corresponding to any images
        nb_false_indexes = 0

        for idx_frame in range(self.nb_images):
            # Get the time
            self.time[idx_frame] = (idx_frame + self.begin_frame) / self.fps

            # Register the predicted and the real position of the head
            if len(predictions[idx_frame]) > 0:
                self.pos_predictions[idx_frame] = np.array([self.left_limit + (predictions[idx_frame][0][1] - self.added_pad) / self.scale , self.left_limit + (predictions[idx_frame][0][2] - self.added_pad) / self.scale ])
                self.pos_regression[idx_frame] = np.array([self.left_limit + (predictions[idx_frame][0][3] - self.added_pad) / self.scale] )
                self.pos_real[idx_frame] = self.left_limit + (labels[idx_frame - nb_false_indexes] - self.added_pad) / self.scale
            else:
                nb_false_indexes += 1

    def save_graphic(self, path_save):
        """
        Save the graphic.

        Args:
            path_save (WindowsPath): the path where the graphic will be saved.
        """
        # Set the figure
        (fig, ax) = plt.subplots()

        # Set the real position
        ax.plot(self.pos_real, self.time, label="real position", c="black")

        # Set the mean for the predictions
        ax.plot(self.pos_regression, self.time, label="estimated position", c="red")

        # Set the uncertainty
        ax.fill_betweenx(self.time, self.pos_predictions[:, 0], self.pos_predictions[:, 1], alpha=0.3, facecolor="blue", label="likely zone")

        # Set the legend and plot it
        ax.set_xlabel("Distance in meters")
        ax.set_ylabel("Time in seconds")
        ax.legend()
        plt.savefig(path_save)
        plt.close()
