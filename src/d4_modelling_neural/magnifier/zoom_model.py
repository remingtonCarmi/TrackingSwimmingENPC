"""
This class is the MODEL of the magnifier.
"""
from pathlib import Path
import numpy as np

# For the MODEL
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    ReLU
)

# To test the MODEL
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader
from src.d4_modelling_neural.magnifier.slice_lane.slice_lanes import slice_lanes


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


class ZoomModel(Model):
    def __init__(self):
        super(ZoomModel, self).__init__()
        self.c32 = Conv2D(32, kernel_size=(4, 6), strides=2, padding="valid", activation="relu")
        self.relu32 = ReLU()

        self.max_poolc32 = MaxPooling2D(pool_size=2, strides=2, padding="valid")

        self.c64 = Conv2D(64, kernel_size=(4, 3), strides=(3, 2), padding="valid", activation="relu")
        self.relu64 = ReLU()

        self.max_poolc64 = MaxPooling2D(pool_size=(4, 6), strides=(4, 6), padding="valid")

        self.flatten = Flatten()
        self.dense = Dense(3, activation="relu")

    def call(self, inputs):
        # Size, without batch size
        # 110 x WindowSize x 3
        x = self.c32(inputs)
        x = self.relu32(x)
        # 54 x 98 x 32
        x = self.max_poolc32(x)

        # 27 x 49 x 32
        x = self.c64(x)
        x = self.relu64(x)
        # 8 x 24 x 64
        x = self.max_poolc64(x)

        # 2 x 4 x 64
        x = self.flatten(x)
        # 512
        x = self.dense(x)
        # 3

        return x


if __name__ == "__main__":
    # - To compute the dimensions - #
    # (input_size, kernel_size, stride, padding, dilation)
    # print(compute_dimension(49, 3, 2, 0, 1))

    # - To try the MODEL - #
    # Parameters
    WINDOW_SIZE = 200
    RECOVERY = 100

    # Paths to data
    PATHS_LABEL = [Path("../../../data/2_processed_positions/tries/vid0.csv")]
    START_DATA_PATHS = Path("../../../data/1_intermediate_top_down_lanes/LANES/tries")
    START_CALIB_PATHS = Path("../../../data/1_intermediate_top_down_lanes/calibration/tries")

    # Generate and load the data
    TRAIN_DATA = generate_data(PATHS_LABEL, starting_data_paths=START_DATA_PATHS, starting_calibration_paths=START_CALIB_PATHS)
    TRAIN_SET = DataLoader(TRAIN_DATA, scale=35, batch_size=1, data_augmenting=False)

    # Slice the first batch
    (LANES, LABELS) = TRAIN_SET[0]
    (SUB_LANES, SUB_LABELS) = slice_lanes(LANES, LABELS, WINDOW_SIZE, RECOVERY)

    model = ZoomModel()
    model.trainable = False
    output = model(SUB_LANES)
    model.summary()
    print(output)
