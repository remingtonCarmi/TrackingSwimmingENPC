from pathlib import Path
import random as rd
import numpy as np
import cv2
from data_generation.transformations.transformations import transform_image, transform_label, augmenting
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(Sequence):
    def __init__(self, data, access_path=None, batch_size=2, nb_classes=10, image_size=1920, for_train=True, data_augmenting=False):
        """
        data = list of [image_path, 'x_head', 'y_head']
        if for_train == True : data augmenting is performed.
        """
        self.data_manager = ImageDataGenerator()
        self.samples = data[:, 0]
        self.labels = data[:, 1:]
        self.shuffle_data()
        self.batch_size = batch_size
        if access_path is None:
            self.access_path = Path("../../output/test/vid1/")
        else:
            self.access_path = access_path
        self.nb_classes = nb_classes
        self.image_size = image_size
        self.for_train = for_train
        self.data_augmenting = data_augmenting

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_path = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size].astype(float)

        length_batch = len(batch_path)
        batch_img = []
        batch_labs = []

        # Get the images
        for idx_img in range(length_batch):
            # Get the image and transform it
            image_path = self.access_path / batch_path[idx_img]
            image_batch = transform_image(image_path)
            label_batch = transform_label(batch_labels[idx_img], self.nb_classes, self.image_size)

            # Perform data augmenting
            if self.data_augmenting:
                random_augmenting = np.random.randint(0, 2)
            else:
                random_augmenting = 0
            (augmented_image, augmented_label) = augmenting(image_batch, label_batch, random_augmenting)

            # Fill the list
            batch_img.append(augmented_image)
            batch_labs.append(augmented_label)

        return np.array(batch_img, dtype=np.float32), np.array(batch_labs, dtype=np.float32)

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
    PERCENTAGE = [0.5, 0.5]
