"""
This module contains the class PredictionMemories.
"""
from pathlib import Path
import numpy as np
import cv2

# To get the left limit and the video length
from src.d7_visualization.tools.get_meters_video import get_meters_video


class PredictionMemories:
    """
    This class collects all the predictions and add them to the original video clip.
    """

    def __init__(
        self,
        begin_time,
        end_time,
        path_video,
        starting_calibration_path,
        dimensions,
        scale,
        extract_image,
        generate_data,
        DataLoader,
        read_homography,
        get_original_image,
    ):
        """
        Construct the list of all the predictions.

        Args:
            begin_time (integer): the beginning time.

            end_time (integer): the ending time.

            path_video (WindowsPath): the path that leads to the video.

            starting_calibration_path (WindowsPath): the path that leads to the calibration path.

            dimensions (list of integers): the dimensions of the transformed images.

            scale (integer): the number of pixels per meters of the transformed images.

            extract_image (function): extract the images.

            generate_data (function): generate the wanted data.

            DataLoader (class): class that loads the data.

            read_homography (function): read a registered homography from a path.

            get_original_image (function): get the original image from a top-view image and an homography.
        """
        # --- General variables --- #
        video = cv2.VideoCapture(str(path_video))
        self.fps = video.get(cv2.CAP_PROP_FPS)

        self.begin_time = begin_time
        self.end_time = end_time
        self.begin_frame = int(begin_time * self.fps)
        self.end_frame = int(end_time * self.fps)

        self.path_video = path_video
        self.video_name = path_video.parts[-1][:-4]

        self.dimensions = dimensions
        self.scale = scale

        self.starting_calibration_path = starting_calibration_path
        self.calibration_path = starting_calibration_path / "{}.txt".format(path_video.parts[-1][:-4])

        length_video = get_meters_video(self.calibration_path)[1]
        self.added_pad = (dimensions[1] - int(scale * length_video)) // 2

        # --- For the original video --- #
        self.extract_image = extract_image

        # Get the original dimensions of the original video clip
        self.original_dimensions = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))]

        self.unscaled_factor = self.original_dimensions[1] / (length_video * scale)

        # To undo the perspective
        self.homography = read_homography(self.calibration_path)
        self.get_original_image = get_original_image

        # --- For the lane video and the graphic --- #
        self.generate_data = generate_data
        self.data_loader = DataLoader
        self.data = None

        # --- To collect the predictions --- #
        self.preds = [[] for frame in range(self.end_frame - self.begin_frame)]

    def in_time(self, data):
        """
        Select the lanes that are after the beginning frame and before the ending frame.

        Args:
            data (list): list of [path_to_image, y_head, x_head, swimming_way, video_length].

        Returns:
            (list): list of [path_to_image, y_head, x_head, swimming_way, video_length].
                Only the interesting elements are kept.
        """
        # Get the frame numbers
        print("One frame number", int(data[0][0].parts[-1][:-4].split("f")[1]))
        frame_numbers = np.array([int(element[0].parts[-1][:-4].split("f")[1]) for element in data])
        print("Frame numbers", frame_numbers)
        # Get the indexes separately
        index_selection_low = np.where(frame_numbers >= self.begin_frame)[0]
        index_selection_high = np.where(frame_numbers < self.end_frame)[0]
        print("Index inter", list(set(index_selection_low).intersection(index_selection_high)))
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
        lane_number = int(frame_name.split("f")[0][1:-1])

        self.preds[frame_number - self.begin_frame].append(
            [lane_number, index_pred_left, index_pred_rigth, index_regression_pred]
        )

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
            original_frames[idx_frame] += self.merge_preds_original(idx_frame)
            np.clip(original_frames[idx_frame], 0, 255, original_frames[idx_frame])

        return original_frames.astype(np.uint8)

    def merge_preds_original(self, idx_frame):
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
            # Vertical rescaling
            vertical_pixel_begin = int(lane_number * self.original_dimensions[0] / 10)
            vertical_pixel_end = int((lane_number + 1) * self.original_dimensions[0] / 10)

            # Horizontal rescaling
            origin_index_left = int((index_pred_left - self.added_pad) * self.unscaled_factor)
            origin_index_rigth = int((index_pred_right - self.added_pad) * self.unscaled_factor)
            origin_index_regression = int((index_regression - self.added_pad) * self.unscaled_factor)

            transformed_image[vertical_pixel_begin:vertical_pixel_end, origin_index_left:origin_index_rigth] += 40

            # [-255, -255, 255] to be sure that after the clip, it will be [0, 0, 255]
            transformed_image[
                vertical_pixel_begin:vertical_pixel_end,
                np.clip(origin_index_regression, 0, self.original_dimensions[1] - 1),
            ] = [-255, -255, 255]

        return self.get_original_image(transformed_image, self.homography, self.original_dimensions)

    def get_lanes(self, lane_number, tries):
        """
        Get the lanes with the predictions.

        Args:
            lane_number (integer): the number of the lane.

            tries (string): indicates if it is a real run.

        Returns:
            list_lanes (list of array): list of the lanes.
        """
        # Get the lanes
        path_label = [Path("data/3_processed_positions{}/{}.csv".format(tries, self.video_name))]
        starting_data_path = Path("data/2_intermediate_top_down_lanes/lanes{}".format(tries))

        self.data = self.generate_data(
            path_label, starting_data_path, self.starting_calibration_path, take_all=True, lane_number=lane_number
        )
        self.data = self.in_time(self.data)
        set_visu = self.data_loader(
            self.data,
            scale=self.scale,
            batch_size=1,
            dimensions=self.dimensions,
            standardization=False,
            augmentation=False,
            flip=False,
        )
        list_lanes = np.zeros((len(self.data), self.dimensions[0], self.dimensions[1], 3))

        # For each image
        for (idx_lane, batch) in enumerate(set_visu):
            (lanes, labels) = batch
            lane = lanes[0]
            frame_name = self.data[idx_lane, 0].parts[-1][:-4]
            frame_number = int(frame_name.split("f")[1])

            list_lanes[idx_lane] = self.merge_preds_lane(frame_number - self.begin_frame, lane)

        return list_lanes.astype(np.uint8)

    def merge_preds_lane(self, idx_frame, lane):
        """
        Add the predictions to the lanes.

        Args:
            idx_frame (integer): the index of the frame.

            lane (array): the lane.

        Returns:
            (array): the lane with the prediction.
        """
        # Modify the color of the lane
        # Classification
        lane[:, self.preds[idx_frame][0][1] : self.preds[idx_frame][0][2]] += 40

        # Regression
        lane[:, self.preds[idx_frame][0][3]] = [0, 0, 255]

        return np.clip(lane, 0, 255)
