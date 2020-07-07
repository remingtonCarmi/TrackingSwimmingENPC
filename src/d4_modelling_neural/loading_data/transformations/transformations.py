from pathlib import Path
import numpy as np
import cv2

# To standardize, scale and pad the images
from src.d4_modelling_neural.loading_data.transformations.transformation_tools.image_transformation import standardize, rescale
from src.d4_modelling_neural.loading_data.transformations.transformation_tools.image_transformation import fill_with_black
import collections


class ImageMagnifier:  # (collections.abc.Sequence):
    def __init__(self, path_image, window_size, recovery, scale, length_video, dimensions):
        # Parameters
        self.window_size = window_size
        self.recovery = recovery
        self.dimensions = dimensions

        # Load, rescale and pad the image
        image = cv2.imread(str(path_image))
        self.image = rescale(image, scale, length_video)
        self.image = fill_with_black(self.image, dimensions)

        # Get the number of sub-images
        self.nb_sub_images = len(self)

    def __len__(self):
        nb_stake = int(np.ceil(self.dimensions[1] / (self.window_size - self.recovery)))
        return nb_stake - self.window_size // (self.window_size - self.recovery) + 1

    def __getitem__(self, idx):
        if idx < self.nb_sub_images - 1:
            pixel_step = self.window_size - self.recovery
            print("Starting point", idx * pixel_step)
            print("Ending point", idx * pixel_step + self.window_size)
            return self.image[:, idx * pixel_step: idx * pixel_step + self.window_size]
        elif idx == self.nb_sub_images - 1:
            print("Starting point", self.dimensions[1] - self.window_size)
            print("Ending point", self.dimensions[1])
            return self.image[:, - self.window_size:]
        else:
            raise StopIteration


if __name__ == "__main__":
    # Parameters
    PATH_IMAGE = Path("../../../../data/1_intermediate_top_down_lanes/lanes/tries/vid1/l1_f0297.jpg")
    WINDOW_SIZE = 200
    RECOVERY = 100
    SCALE = 35
    LENGTH_VIDEO = 25
    DIMENSIONS = [110, 1820]

    IMAGE_MAGNIFIER = ImageMagnifier(PATH_IMAGE, WINDOW_SIZE, RECOVERY, SCALE, LENGTH_VIDEO, DIMENSIONS)

    print("Nb sub images", len(IMAGE_MAGNIFIER))

    for (idx_image, sub_image) in enumerate(IMAGE_MAGNIFIER):
        print("Image n° {}".format(idx_image))
        print("Dimensions", sub_image.shape)
        cv2.imshow("Image n' {}".format(idx_image), sub_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
