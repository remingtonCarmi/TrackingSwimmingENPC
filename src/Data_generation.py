""" This code aims to generate more images from one to increase data"""

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.load_data import load_data
from src.head_coords_image_association import associate_head_coordinates_with_image


def convert_into_image_tensor(img, coord):
    """
    Enables to convert an image in the right format with tensorflow

    :param
        image: (array) path in which we stored the image
    :return:
        image: (tensor) image in the right format
        coords_h: (tuple) contains the head coordinates
    """
    img = np.asarray(img, np.float32)
    # image = tf.pack(image)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, coord


def augment(coord, image):
    """
    Augments the data: produces 6 images with one input image

    """
    image, coord = convert_into_image_tensor(image, coord)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image1 = tf.image.random_flip_left_right(image)
    image2 = tf.image.random_saturation(image, 0, 3)
    image3 = tf.image.random_brightness(image, max_delta=0.5)
    image4 = tf.image.random_saturation(image, 1, 2)
    image5 = tf.image.random_brightness(image, max_delta=0.3)
    # image6 = tf.image.translate(image, [3,3])
    pairs = [[image, coord], [image1, coord], [image2, coord], [image3, coord], [image4, coord], [image5, coord]]

    return pairs


if __name__ == "__main__":
    print('ok')
    IMAGE_PATH = Path("../data/lanes/f123_l4.jpg")
    DATA_PATH = Path("../data/head_points/test_file.csv")
    data, coords = load_data(DATA_PATH)
    image, coords_h = associate_head_coordinates_with_image(IMAGE_PATH, data)
    list_pairs = augment(coords_h, image)
    sess = tf.InteractiveSession()
    image = list_pairs[0][0].eval()
    print(image.shape)
    image_p = [list(map(int, p)) for p in image[:][:]]
    print(image[0][0])
    plt.figure()
    plt.imshow(image)
    cv2.imshow(cv2.WINDOW_NORMAL, image)
    sess.close()
