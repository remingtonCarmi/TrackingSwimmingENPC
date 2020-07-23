"""
This class is the model of the magnifier.
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
    BatchNormalization,
    ReLU
)

# To test the MODEL
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader
from src.d5_model_evaluation.slice_lane.slice_lanes import slice_lane


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
    def __init__(self, close_to_head):
        """
        Construct the function that will be applied.

        Args:
            close_to_head (boolean): says if the images will be small.
        """
        super(ZoomModel, self).__init__()
        if not close_to_head:
            self.c32 = Conv2D(32, kernel_size=(4, 6), strides=2, padding="valid")
            self.batch_norm32 = BatchNormalization()
            self.relu32 = ReLU()

            self.max_poolc32 = MaxPooling2D(pool_size=2, strides=2, padding="valid")

            self.c64 = Conv2D(64, kernel_size=(4, 3), strides=(3, 2), padding="valid")
            self.batch_norm64 = BatchNormalization()
            self.relu64 = ReLU()

            self.max_poolc64 = MaxPooling2D(pool_size=(4, 6), strides=(4, 6), padding="valid")

            self.flatten = Flatten()
            self.dense = Dense(3)

        else:
            self.c32 = Conv2D(32, kernel_size=3, strides=1, padding="valid")
            self.batch_norm32 = BatchNormalization()
            self.relu32 = ReLU()

            self.max_poolc32 = MaxPooling2D(pool_size=3, strides=3, padding="valid")

            self.c64 = Conv2D(64, kernel_size=3, strides=2, padding="valid")
            self.batch_norm64 = BatchNormalization()
            self.relu64 = ReLU()

            self.max_poolc64 = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding="valid")

            self.flatten = Flatten()
            self.dense = Dense(3)

    def call(self, inputs):
        # Size, without batch size // close_to_head
        # 108 x WindowSize x 3 // 108 x WindowSize x 3
        x = self.c32(inputs)
        x = self.batch_norm32(x, self.trainable)
        x = self.relu32(x)
        # 54 x 98 x 32 // 106 x 28 x 32
        x = self.max_poolc32(x)

        # 27 x 49 x 32 // 35 x 9 x 32
        x = self.c64(x)
        x = self.batch_norm64(x, self.trainable)
        x = self.relu64(x)
        # 8 x 24 x 64 // 17 x 4 x 64
        x = self.max_poolc64(x)

        # 2 x 4 x 64 // 4 x 2 x 64
        x = self.flatten(x)
        # 512 // 512
        x = self.dense(x)
        # 3 // 3

        return x


if __name__ == "__main__":
    # - To compute the dimensions - #
    # (input_size, kernel_size, stride, padding, dilation)
    # print(compute_dimension(49, 3, 2, 0, 1))

    # - To try the MODEL - #
    # Parameters
    WINDOW_SIZE = 30
    RECOVERY = 0
    CLOSED_TO_HEAD = True

    # Paths to data
    PATHS_LABEL = [Path("../../data/3_processed_positions/tries/vid0.csv")]
    START_DATA_PATHS = Path("../../data/2_intermediate_top_down_lanes/lanes/tries")
    START_CALIB_PATHS = Path("../../data/2_intermediate_top_down_lanes/calibration/tries")

    # Generate and load the data
    TRAIN_DATA = generate_data(PATHS_LABEL, starting_data_paths=START_DATA_PATHS, starting_calibration_paths=START_CALIB_PATHS)
    TRAIN_SET = DataLoader(TRAIN_DATA, scale=35, batch_size=1)

    # Slice the first batch
    (LANES, LABELS) = TRAIN_SET[0]
    (SUB_LANES, SUB_LABELS, LANE_ITERATOR) = slice_lane(LANES[0], LABELS[0], WINDOW_SIZE, RECOVERY)

    model = ZoomModel(CLOSED_TO_HEAD)
    model.trainable = False
    output = model(SUB_LANES)
    model.summary()
    print(output)
