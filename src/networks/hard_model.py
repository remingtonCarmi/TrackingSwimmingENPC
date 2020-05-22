from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
    ReLU,
    Dropout,
    Softmax,
    GlobalAveragePooling2D
)


class HardModel(Model):
    def __init__(self, nb_classes=10):
        super(HardModel, self).__init__()
        self.c32 = Conv2D(32, kernel_size=(7, 9), strides=2, padding="valid")
        self.batch_norm32 = BatchNormalization()
        self.relu32 = ReLU()

        self.max_pool32 = MaxPooling2D(pool_size=2, strides=2, padding="valid")

        self.c64 = Conv2D(64, kernel_size=(5, 7), strides=(1, 2), padding="valid")
        self.batch_norm64 = BatchNormalization()
        self.relu64 = ReLU()

        self.max_pool64 = MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="valid")

        self.c128 = Conv2D(128, kernel_size=(5, 7), strides=(1, 2), padding="valid", )
        self.batch_norm128 = BatchNormalization()
        self.relu128 = ReLU()

        self.max_pool128 = MaxPooling2D(pool_size=2, strides=2, padding="valid")

        self.gap = GlobalAveragePooling2D()

        self.dropout = Dropout(0.1)
        self.dense = Dense(nb_classes, activation="relu")
        self.soft_max = Softmax()

    def call(self, inputs):

        # Size, without batch size
        # First layer
        # 108 x 1920 x 3
        x = self.c32(inputs)
        # 51 x 956 x 32
        x = self.batch_norm32(x)
        x = self.relu32(x)

        # 51 x 956 x 32
        x = self.max_pool32(x)
        # 25 x 478 x 32

        # Second layer
        # 25 x 478 x 32
        x = self.c64(x)
        # 21 x 236 x 64
        x = self.batch_norm64(x)
        x = self.relu64(x)

        # 21 x 236 x 64
        x = self.max_pool64(x)
        # 10 x 59 x 64

        # Third layer
        # 10 x 59 x 64
        x = self.c128(x)
        # 6 x 27 x 64
        x = self.batch_norm128(x)
        x = self.relu128(x)

        # 6 x 27 x 64
        x = self.max_pool128(x)
        # 3 x 13 x 128

        # 3 x 13 x 128
        x = self.gap(x)
        # 128

        if self.trainable:
            x = self.dropout(x)

        # Fourth layer
        # 128
        x = self.dense(x)
        # number classes
        x = self.soft_max(x)

        return x


if __name__ == "__main__":
    PATH_DATA = Path("../../output/test/vid1/")
    PATH_LABEL = Path("../../output/test/vid1.csv")
    PERCENTAGE = 0.5
    NB_CLASSES = 10

    # Generate and load the data
    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
    TRAIN_SET = GENERATOR.train
    TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, nb_classes=NB_CLASSES)

    BATCH = np.array(TRAIN_DATA[0][0])

    model = HardModel(nb_classes=NB_CLASSES)

    output = model(BATCH)
    model.summary()

    print(output)
