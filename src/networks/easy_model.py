from data_generation.data_loader import DataLoader
from data_generation.data_generator import DataGenerator

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    MaxPooling3D,
    Input,
    Reshape,
    Lambda
)


def compute_dimension(input_size, kernel_size, stride, padding, dilation):
    """
    Computes the output size of an image that goes through a layer.

    Args:
        input_size (int): the size of the input.

        kernel_size (int): the size of the layer's kernel.

        stride (int): the stride of the layer.

        padding (int): the size of the layer's padding.

        dilation (int): the dilation of the layer.

    Returns:
        output_size (int): the size of the output.
    """
    return np.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class EasyModel(Model):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.c32 = Conv2D(32, kernel_size=(3, 3), strides=3, padding="valid", activation="relu")
        self.max_poolc32 = MaxPooling2D(pool_size=(4, 5), strides=(4, 5), padding="valid")
        self.c64 = Conv2D(64, kernel_size=(3, 3), strides=3, padding="valid", activation="relu")
        self.max_poolc64 = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), padding="valid")
        self.flatten = Flatten()
        self.dense10 = Dense(10, activation="relu")

    def call(self, inputs):

        # Size, without batch size
        # 108 x 1920 x 3
        x = self.c32(inputs)
        # 36 x 640 x 32
        x = self.max_poolc32(x)
        # 9 x 128 x 32
        x = self.c64(x)
        # 3 x 42 x 64
        x = self.max_poolc64(x)
        # 1 x 21 x 64
        x = self.flatten(x)
        # 1344
        x = self.dense10(x)
        # 10

        return x


if __name__ == "__main__":
    PATH_DATA = Path("../output/test/vid1/")
    PATH_LABEL = Path("../output/test/vid1.csv")
    POURCENTAGE = [0.5, 0.5]

    # (input_size, kernel_size, stride, padding, dilation)
    # print(compute_dimension(42, 2, 2, 0, 1))

    # Generate and load the data
    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, pourcentage=POURCENTAGE)
    TRAIN_SET = GENERATOR.train
    TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA)

    BATCH = np.array(TRAIN_DATA[0][0])

    model = EasyModel()

    output = model(BATCH)
    print(output)
