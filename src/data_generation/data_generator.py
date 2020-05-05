from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np
import random as rd


def get_frame_lane(name_file):
    frame_lane = name_file[1: -4].split("_")
    return int(frame_lane[0][1:]), int(frame_lane[1][1:])


class DataGenerator:
    def __init__(self, path_data, path_label, pourcentage=[0.8, 0.2]):
        # Data and labels
        self.data = listdir(path_data)
        self.labels = pd.read_csv(path_label)

        # Manage the numbers of samples
        self.nb_samples = len(self.data)
        self.nb_trains = int(self.nb_samples * pourcentage[0])
        self.nb_valids = self.nb_samples - self.nb_trains

        # Get the full data_set
        self.full_data = [["path", 0, 0] for image in range(self.nb_samples)]

        # Training and validation sets
        self.train = [["path", 0, 0] for image in range(self.nb_trains)]
        self.valid = [["path", 0, 0] for image in range(self.nb_valids)]

        # Fill the sets
        self.fill_full_data()
        self.fill_sets()

    def get_label(self, path):
        (frame, lane) = get_frame_lane(path)
        if (frame, lane) in self.labels.index:
            return self.labels.loc[[(frame, lane)]].to_numpy()[0]
        else:
            return np.array([-2, -2])

    def fill_full_data(self):
        for idx_sample in range(self.nb_samples):
            path_sample = self.data[idx_sample]
            self.full_data[idx_sample][0] = path_sample
            self.full_data[idx_sample][1:] = self.get_label(path_sample)

    def fill_sets(self):
        rd.shuffle(self.full_data)
        self.train = np.array(self.full_data[: self.nb_trains])
        self.valid = np.array(self.full_data[: self.nb_valids])


if __name__ == "__main__":
    # EXCEPTION CLASS TO MAKE :
    # THE INPUT PATHS DO NOT EXIST
    # THE INPUT PATHS DO NOT MATCH
    # THE PATH_DATA IS EMPTY
    PATH_DATA = Path("../../output/test/vid1/")
    PATH_LABEL = Path("../../output/test/vid1.csv")
    POURCENTAGE = [0.5, 0.5]

    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
    print(GENERATOR.train)
