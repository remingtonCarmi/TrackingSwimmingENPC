from pathlib import Path
import numpy as np
import cv2


class ImageMagnifier:
    def __init__(self, lane, label, window_size, recovery):
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
        nb_stake = int(np.ceil(self.dimensions[1] / (self.window_size - self.recovery)))
        return nb_stake - self.window_size // (self.window_size - self.recovery) + 1

    def __getitem__(self, idx):
        if idx < self.nb_sub_images - 1:
            pixel_step = self.window_size - self.recovery

            # Compute the label
            if idx * pixel_step <= self.label[1] <= idx * pixel_step + self.window_size:
                present = True
                column = self.label[1] - idx * pixel_step
            else:
                present = False
                column = -1

            print("Starting point", idx * pixel_step)
            print("Ending point", idx * pixel_step + self.window_size)
            return self.lane[:, idx * pixel_step: idx * pixel_step + self.window_size], [present, column]
        elif idx == self.nb_sub_images - 1:
            # Compute the label
            if self.dimensions[1] - self.window_size <= self.label[1] <= self.dimensions[1]:
                present = True
                column = self.label[1] - self.dimensions[1] - self.window_size
            else:
                present = False
                column = -1

            print("Starting point", self.dimensions[1] - self.window_size)
            print("Ending point", self.dimensions[1])
            return self.lane[:, - self.window_size:], [present, column]
        else:
            raise StopIteration


if __name__ == "__main__":
    # Data
    # PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0297.jpg")
    PATH_IMAGE = Path("../../../../data/4_model_output/tries/scaled_images/scaled_l1_f0275.jpg")

    LANE = cv2.imread(str(PATH_IMAGE))
    # LABEL = np.array([54, 950])
    LABEL = np.array([49, 648])
    WINDOW_SIZE = 200
    RECOVERY = 100

    IMAGE_MAGNIFIER = ImageMagnifier(LANE, LABEL, WINDOW_SIZE, RECOVERY)

    print("Nb sub image", len(IMAGE_MAGNIFIER))

    for (idx_image, (sub_lane, sub_label)) in enumerate(IMAGE_MAGNIFIER):
        print("Image n° {}. Present = {}".format(idx_image, sub_label[0]))
        print("Dimensions", sub_lane.shape)
        if sub_label[0]:
            sub_lane[:, sub_label[1]] = [0, 0, 255]
        cv2.imshow("Image n' {}. Present = {}".format(idx_image, sub_label[0]), sub_lane)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
