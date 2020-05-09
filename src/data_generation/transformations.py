import cv2
import numpy as np


def transform(image_path):
    image = cv2.imread(str(image_path))
    return normalize(image)


def normalize(image):
    """
    Normalize between -1 and 1.
    """
    min_image = np.min(image)
    return 2 * (image - min_image) / (np.max(image) - min_image) - 1

