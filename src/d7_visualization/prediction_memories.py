"""
This module contains the class PredictionMemories.
"""
import numpy as np
import cv2

# To get the left limit and the video length
from src.d7_visualization.tools.get_meters_video import get_meters_video


class PredictionMemories:
    """
    This class collects all the predictions and add them to the original video clip.
    """
    def __init__(self, begin_time, end_time, path_video, read_homography, get_original_image, starting_calibration_path, dimensions, extract_image, scale):
        """
        Construct the list of all the predictions.
        """
        # Get the fps and the original dimensions of the original video clip
        video = cv2.VideoCapture(str(path_video))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.original_dimensions = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))]

        # About the original video
        self.begin_time = begin_time
        self.end_time = end_time
        self.begin_frame = int(begin_time * self.fps)
        self.end_frame = int(end_time * self.fps)
        self.path_video = path_video
        self.video_name = path_video.parts[-1][:-4]
        self.extract_image = extract_image

        # Get the added_pad to transfer the transformed lane to the original lane
        self.calibration_path = starting_calibration_path / "{}.txt".format(path_video.parts[-1][:-4])
        length_video = get_meters_video(self.calibration_path)[1]
        self.added_pad = (dimensions[1] - int(scale * length_video)) // 2
        self.unscaled_factor = self.original_dimensions[1] / (length_video * scale)

        # To undo the perspective
        self.homography = read_homography(self.calibration_path)
        self.get_original_image = get_original_image
        self.dimensions = dimensions

        # To collect the predictions
        self.preds = [[] for frame in range(self.end_frame - self.begin_frame)]

    def in_time(self, data):
        """
        Select the lanes that are after the beginning frame and before the ending frame.

        Args:
            data (list): list of [path_to_image, y_head, x_head, swimming_way, video_length].
        .
        Returns:
            (list): list of [path_to_image, y_head, x_head, swimming_way, video_length].
                Only the interesting elements are kept.
        """
        # Get the frame numbers
        frame_numbers = np.array([int(element[0].parts[-1][:-4].split("f")[1]) for element in data])

        # Get the indexes separately
        index_selection_low = np.where(frame_numbers >= self.begin_frame)[0]
        index_selection_high = np.where(frame_numbers < self.end_frame)[0]

        # Get the intersections of the indexes
        return data[list(set(index_selection_low).intersection(index_selection_high))]

    def update(self, frame_name, index_pred_left, index_pred_rigth, index_regression_pred):
        """
        Add the prediction to the list that collects them.

        Args:
            frame_name (string): the name of the image.

            index_pred_left (integer): the index of the left limit of the predicted zone in the transformed image.

            index_pred_rigth (integer): the index of the right limit of the predicted zone in the transformed image.

            index_regression_pred (integer): the index of the predicted head in the transformed image.
        """
        # Get the frame and lane number
        frame_number = int(frame_name.split("f")[1])
        lane_number = int(frame_name.split("f")[0][1: -1])

        # Put the index in the original image coordinate
        origin_index_left = int((index_pred_left - self.added_pad) * self.unscaled_factor)
        origin_index_rigth = int((index_pred_rigth - self.added_pad) * self.unscaled_factor)
        origin_index_regression = int((index_regression_pred - self.added_pad) * self.unscaled_factor)

        self.preds[frame_number - self.begin_frame].append([lane_number, origin_index_left, origin_index_rigth, origin_index_regression])

    def get_original_frames(self):
        """
        Get the original images with the predictions.

        Returns:
            original_frames (4d-array): the list of the images
        """
        # Extract the original images
        original_frames = np.array(self.extract_image(self.path_video, self.begin_time, self.end_time), dtype=float)
        print("preds", self.preds)
        # For each frame, we add the prediction
        for idx_frame in range(self.end_frame - self.begin_frame):
            original_frames[idx_frame] += self.merge_preds(idx_frame)
            np.clip(original_frames[idx_frame], 0, 255, original_frames[idx_frame])

        return original_frames.astype(np.uint8)

    def merge_preds(self, idx_frame):
        """"
        Merge the predictions for the idx_frame^th frame.

        Args:
            idx_frame (integer): the index of the frame minus the first frame index.

        Returns:
            (3d-array): the image of the prediction in the original video clip point of view.
        """
        # Initialise with a black picture
        transformed_image = np.zeros((self.original_dimensions[0], self.original_dimensions[1], 3))

        # For each lane, we add the prediction
        for (lane_number, index_pred_left, index_pred_right, index_regression) in self.preds[idx_frame]:
            vertical_pixel_begin = int(lane_number * self.original_dimensions[0] / 10)
            vertical_pixel_end = int((lane_number + 1) * self.original_dimensions[0] / 10)
            transformed_image[vertical_pixel_begin: vertical_pixel_end, index_pred_left: index_pred_right] += 40

            # [-255, -255, 255] to be sure that after the clip, it will be [0, 0, 255]
            transformed_image[vertical_pixel_begin: vertical_pixel_end, np.clip(index_regression, 0, self.original_dimensions[1] - 1)] = [-255, -255, 255]

        return self.get_original_image(transformed_image, self.homography, self.original_dimensions)
