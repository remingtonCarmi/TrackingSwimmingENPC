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


def augment_data(coord, image):
    """
    Augments the data: produces 6 images with one input image

    """
    h, w = image.shape[0], image.shape[1]
    img, coord = convert_into_image_tensor(image, coord)
    img = tf.image.convert_image_dtype(img, tf.float32)
    image1 = tf.image.flip_left_right(img)
    image2 = tf.image.random_saturation(img, 0, 3)
    image3 = tf.image.random_brightness(img, max_delta=0.5)
    image4 = tf.image.random_saturation(img, 1, 2)
    image5 = tf.image.random_brightness(img, max_delta=0.3)
    # image6 = tf.image.translate(image, [3,3])
    pairs = [[img, coord], [image1, (h-coord[0], w-coord[1])], [image2, coord], [image3, coord], [image4, coord], [image5, coord]]

    return pairs


if __name__ == "__main__":
    print('ok')
    DATA_PATH = Path("../output/test/test_file.csv")
    IMAGE_PATH = Path("../output/test/vid0/f4_c3.jpg")
    data, coords = load_data(DATA_PATH)
    image, coords_h = associate_head_coordinates_with_image(IMAGE_PATH, data)
    list_pairs = augment_data(coords_h, image)
    sess = tf.InteractiveSession()
    for i in range(len(list_pairs)):
        image = list_pairs[i][0].eval()
        print(image.shape)
        image_p = np.int_(image)
        b, g, r = cv2.split(image_p)  # get b,g,r
        image_p = cv2.merge([r, g, b])
        plt.figure()
        plt.imshow(image_p)
        plt.show()
    sess.close()
