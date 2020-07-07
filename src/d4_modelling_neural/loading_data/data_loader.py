"""
This module loads the images one by one when the object is called. It can perform data augmenting.
"""
from pathlib import Path
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.transformations.transformations import transform_label, augmenting, standardize
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(Sequence):
    """
    The class to load the data.
    """
    def __init__(self, data, batch_size=2, window_size=100, recovery=0, scale=35, data_augmenting=False):
        """
        Create the loader.

        Args:
            data (list of 4 : [WindowsPath, integer, integer, float):
                List of [image_path, 'x_head', 'y_head', 'length_video]

            batch_size (integer): the size of the batches
                Default value = 2

            nb_classes (integer): the
        """
        # To transform the images
        self.data_manager = ImageDataGenerator()

        # The data
        self.samples = data[:, 0]
        self.labels = data[:, 1: -1]
        self.lengths_video = data[:, -1]

        # The parameters
        self.batch_size = batch_size
        self.data_augmenting = data_augmenting

        # The magnifying glass
        self.window_size = window_size
        self.recovery = recovery
        self.scale = scale

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        # Get the paths and the labels
        batch_path = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size].astype(float)

        # Get the specific size of the batch
        length_batch = len(batch_path)
        batch_img = []
        batch_labs = []

        # Get the images
        for idx_img in range(length_batch):
            # Get the image and transform it
            image_path = self.access_path / batch_path[idx_img]
            image_batch = cv2.imread(str(image_path))
            label_batch = transform_label(batch_labels[idx_img], self.nb_classes, self.image_size)

            # Perform data augmenting
            if self.data_augmenting:
                random_augmenting = np.random.randint(0, 6)
            else:
                random_augmenting = 3
            (augmented_image, augmented_label) = augmenting(image_batch, label_batch, random_augmenting, self.data_manager, self.nb_classes)
            # Fill the list
            batch_img.append(standardize(augmented_image))
            batch_labs.append(augmented_label)

        return np.array(batch_img, dtype=np.float32), np.array(batch_labs, dtype=np.float32)

    def on_epoch_end(self):
        self.shuffle_data()

    def shuffle_data(self):
        full_data = list(zip(self.samples, self.labels))
        rd.shuffle(full_data)
        (self.samples, self.labels) = zip(*full_data)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)


if __name__ == "__main__":
    PATHS_LABEL = [Path("../../../data/2_processed_positions/tries/vid0.csv")]
    BATCH_SIZE = 1

    # Data generator
    TRAIN_SET = generate_data(PATHS_LABEL, take_all=False)

    # Data loader
    LOADER = DataLoader(TRAIN_SET, data_augmenting=False, batch_size=BATCH_SIZE)

    for (BATCH, LABELS) in LOADER:
        # Get the image
        b, g, r = cv2.split(BATCH[0])  # get b,g,r
        rgb_img = cv2.merge([r, g, b])

        # Show the image
        plt.imshow(rgb_img.astype(int))
        print(LABELS[0])
        plt.show()
