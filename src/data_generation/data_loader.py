from pathlib import Path
import random as rd
import numpy as np
import cv2
from src.data_generation.data_generator import DataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(Sequence):
    def __init__(self, data, access_path=Path("../../output/test/vid1/"), batch_size=2, for_train=True):
        """
        data = list of [image_path, 'x_head', 'y_head']
        if for_train == True : data augmenting is performed.
        """
        self.data_manager = ImageDataGenerator()
        self.samples = data[:, 0]
        self.labels = data[:, 1:]
        self.shuffle_data()
        self.batch_size = batch_size
        self.for_train = for_train
        self.access_path = access_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_path = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size].astype(float)

        length_batch = len(batch_path)
        batch_img = []

        # Get the images
        for idx_img in range(length_batch):
            image_path = self.access_path / batch_path[idx_img]
            batch_img.append(cv2.imread(str(image_path)))

        return batch_img, batch_labels

    def on_epoch_end(self):
        if self.for_train:
            self.shuffle_data()

    def shuffle_data(self):
        full_data = list(zip(self.samples, self.labels))
        rd.shuffle(full_data)
        (self.samples, self.labels) = zip(*full_data)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)


if __name__ == "__main__":
    PATH_DATA = Path("../../output/test/vid1/")
    PATH_LABEL = Path("../../output/test/vid1.csv")
    POURCENTAGE = [0.5, 0.5]

    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
    TRAIN_SET = GENERATOR.train
    LOADER = DataLoader(TRAIN_SET)
    print(LOADER[0])
