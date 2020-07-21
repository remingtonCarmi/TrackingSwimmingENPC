"""
This module contains the GraphicManager class.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

# To get the left limit
from src.d7_visualization.tools.get_meters_video import get_meters_video


class GraphicManager:
    """
    Class to manage the graphic data.
    """
    def __init__(self, path_video, path_calibration, scale, nb_images, horiz_dimension):
        """
        Construct the parameters and the data for the figure.

        Args:
            path_video (WindowsPath): the path to the video.

            path_calibration (WindowsPath): the path to the calibration file.

            scale (integer): the scale of the images in pixels per meter.

            nb_images (integer): the number of images.

            horiz_dimension (integer): the horizontal dimension of the images.
        """
        # Parameters
        video = cv2.VideoCapture(str(path_video))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        (self.left_limit, length_video) = get_meters_video(path_calibration)
        self.scale = scale

        self.added_pad = ((horiz_dimension - int(scale * length_video)) // 2) / scale

        # Data for figure
        self.pos_predictions = np.zeros((nb_images, 2))
        self.pos_real = np.zeros(nb_images)
        self.time = np.zeros(nb_images)
        self.lane_numbers = np.zeros(nb_images)

    def update(self, idx_image, frame_name, index_preds, label):
        """
        Update the lists with a new prediction.

        Args:
            idx_image (integer): the index of the image.

            frame_name (string): the name of the image.

            index_preds (array): the index of the columns where the head should be.

            label (integer): the column where the head is.
        """
        # Update the frame number and the lane number
        frame_number = int(frame_name.split("f")[1])
        self.lane_numbers[idx_image] = int(frame_name.split("f")[0][1: -1])

        # Get the time
        self.time[idx_image] = frame_number / self.fps

        # Register the predicted and the real position of the head
        self.pos_predictions[idx_image] = np.array([self.left_limit + index_preds[0] / self.scale - self.added_pad, self.left_limit + index_preds[-1] / self.scale - self.added_pad])
        self.pos_real[idx_image] = self.left_limit + label / self.scale - self.added_pad

    def make_graphic(self, path_save):
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
        ax.plot((self.pos_predictions[:, 0] + self.pos_predictions[:, 1]) / 2, self.time, label="estimated position", c="blue")

        # Set the uncertainty
        ax.fill_betweenx(self.time, self.pos_predictions[:, 0], self.pos_predictions[:, 1], alpha=0.3, facecolor="blue", label="likely zone")

        # Set the legend and plot it
        ax.set_xlabel("Distance in meters")
        ax.set_ylabel("Time in seconds")
        ax.legend()
        plt.savefig(path_save)
        plt.close()
