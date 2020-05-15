import cv2
import numpy as np


# --- Transformation for the images --- #
def transform_image(image_path):
    image = cv2.imread(str(image_path))
    return standardize(image)


def standardize(image):
    """
    Normalize between -1 and 1.
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


# --- Transformation for the labels --- #
def transform_label(label, nb_classes, image_size):
    length_class = image_size / nb_classes

    return int(label[0] // length_class)


# --- Data augmenting --- #
def augmenting(images, labels, random_seed):
    if random_seed == 0:
        return images, labels


if __name__ == "__main__":
    IMAGE = np.array([[[19, 3, 0], [12, 3, 2]], [[10, 31, 2], [2, 23, 28]]])
    LABEL = np.array([450, 12])

    print(standardize(IMAGE))
    print(transform_label(LABEL, 10, 1000))
