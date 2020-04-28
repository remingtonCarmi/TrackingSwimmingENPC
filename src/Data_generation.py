""" This code aims to generate more images from one to increase data"""

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.load_data import load_data
from src.head_coords_image_association import associate_head_coordinates_with_image


def convert_into_image_tensor(image, coords_h):
    """
    Enables to convert an image in the right format with tensorflow

    :param
        image: (array) path in which we stored the image
    :return:
        image: (tensor) image in the right format
        coords_h: (tuple) contains the head coordinates
    """
    image = np.asarray(image, np.float32)
    #image = tf.pack(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, coords_h


def augment(coords_h, image):
    """
    Augments the data: produces 6 images with one input image

    """

    image, coords_h = convert_into_image_tensor(image, coords_h)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image1 = tf.image.random_flip_left_right(image)
    image2 = tf.image.random_saturation(image, 0, 3)
    image3 = tf.image.random_brightness(image, max_delta=0.5)
    image4 = tf.image.random_saturation(image, 1, 2)
    image5 = tf.image.random_brightness(image, max_delta=0.3)
    #image6 = tf.image.translate(image, [3,3])


    list_pairs = [[image, coords_h], [image1, coords_h], [image2, coords_h], [image3, coords_h], [image4, coords_h], [image5, coords_h]]

    return list_pairs


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


if __name__ == "__main__":
    print('ok')
    IMAGE_PATH = Path("../data/lanes/f123_l4.jpg")
    DATA_PATH = Path("../data/head_points/test_file.csv")
    data, coords = load_data(DATA_PATH)
    image, coords_h = associate_head_coordinates_with_image(IMAGE_PATH, data)
    list_pairs = augment(coords_h, image)
    print(list_pairs[0][0].eval())

