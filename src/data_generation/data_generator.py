from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np
import random as rd


def get_frame_lane(name_file):
    frame_lane = name_file[: -4].split("_")
    return int(frame_lane[0][1:]), int(frame_lane[1][1:])


class DataGenerator:
    def __init__(self, path_data, path_label, percentage=0.8):
        # Data and labels
        self.path_data = path_data
        self.labels = pd.read_csv(path_label)

        # Get the full data_set
        self.full_data = self.fill_full_data()

        # Manage the numbers of samples
        self.nb_samples = len(self.full_data)
        self.nb_trains = int(self.nb_samples * percentage)
        self.nb_valids = self.nb_samples - self.nb_trains

        # Training and validation sets
        (self.train, self.valid) = self.fill_sets()

    def fill_full_data(self):
        full_data = []
        for ((lane, frame), label) in self.labels.iterrows():
            # Add to the list only if the image has been labeled with a right position
            if label[0] >= 0:
                name_image = "l{}_f{}.jpg".format(lane, str(frame).zfill(4))
                path_image = self.path_data / name_image
                # Add to the list only if the image is in the computer
                if path_image.exists():
                    full_data.append([name_image, label[0], label[1]])

        return full_data

    def fill_sets(self):
        return np.array(self.full_data[: self.nb_trains]), np.array(self.full_data[self.nb_trains:])


if __name__ == "__main__":
    # EXCEPTION CLASS TO MAKE :
    # THE INPUT PATHS DO NOT EXIST
    # THE INPUT PATHS DO NOT MATCH
    # THE PATH_DATA IS EMPTY
    PATH_DATA = Path("../../output/test/vid1/")
    PATH_LABEL = Path("../../output/test/vid1.csv")
    PERCENTAGE = 0.5

    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
    print(GENERATOR.train)
    print(GENERATOR.valid)
