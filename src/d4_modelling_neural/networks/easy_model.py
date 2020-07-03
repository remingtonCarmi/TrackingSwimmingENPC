from pathlib import Path
from src.d4_modelling_neural.loading_data.data_generator import DataGenerator
from src.d4_modelling_neural.loading_data.data_loader import DataLoader
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    ReLU,
    Softmax
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
    def __init__(self, nb_classes=10):
        super(EasyModel, self).__init__()
        self.c32 = Conv2D(32, kernel_size=(7, 7), strides=(2, 4), padding="valid", activation="relu")
        self.relu32 = ReLU()

        self.max_poolc32 = MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="valid")

        self.c64 = Conv2D(64, kernel_size=(4, 4), strides=(3, 5), padding="valid", activation="relu")
        self.relu64 = ReLU()

        self.max_poolc64 = MaxPooling2D(pool_size=(4, 5), strides=(4, 5), padding="valid")

        self.flatten = Flatten()
        self.dense = Dense(nb_classes, activation="relu")
        self.soft_max = Softmax()

    def call(self, inputs):

        # Size, without batch size
        # 108 x 1920 x 3
        x = self.c32(inputs)
        x = self.relu32(x)
        # 51 x 479 x 32
        x = self.max_poolc32(x)
        # 25 x 119 x 32
        x = self.c64(x)
        x = self.relu64(x)
        # 8 x 24 x 64
        x = self.max_poolc64(x)
        # 2 x 4 x 64
        x = self.flatten(x)
        # 512
        x = self.dense(x)
        # 10
        if not self.trainable:
            x = self.soft_max(x)
        # 10

        return x


if __name__ == "__main__":
    PATH_DATA = Path("../../../data/1_intermediate_top_down_lanes/lanes/vid0/")
    PATH_LABEL = Path("../../../data/2_processed_positions/vid0.csv")
    PERCENTAGE = 0.5
    NB_CLASSES = 10

    # (input_size, kernel_size, stride, padding, dilation)
    # print(compute_dimension(23, 5, 5, 0, 1))

    # Generate and load the data
    GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
    TRAIN_SET = GENERATOR.train
    TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, nb_classes=5)

    BATCH = np.array(TRAIN_DATA[0][0])

    model = EasyModel(nb_classes=NB_CLASSES)
    model.trainable = False
    output = model(BATCH)
    model.summary()
    print(output)
