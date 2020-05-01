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
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, coord


def augment_data(coord, image):
    """
    Augments the data: produces 6 images with one input image

    """
    h, w = [image.shape[0], image.shape[1]]
    img, coord = convert_into_image_tensor(image, coord)
    img = tf.image.convert_image_dtype(img, tf.float32)
    image1 = tf.image.flip_left_right(img)
    image2 = tf.image.random_saturation(img, 0, 3)
    image3 = tf.image.random_brightness(img, max_delta=0.5)
    image4 = tf.image.random_saturation(img, 1, 2)
    image5 = tf.image.random_brightness(img, max_delta=0.3)
    image6 = tf.image.random_contrast(img, 0, 3)
    pairs = [[img, coord], [image1, (w - coord[0], h - coord[1])], [image2, coord],
             [image3, coord], [image4, coord], [image5, coord],
             [image6, coord]]

    return pairs


class HeadCoordMatchingError(Exception):
    """ Verify if the head coordinates are relevant when we look at the corresponding image"""

    def __init__(self, img, coord):
        """
        Constructs the image's name.
        """
        self.image = img
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.coord_x = coord[0]
        self.coord_y = coord[1]

    def __repr__(self):
        """
        :return: (string) Error message if the coords don't match with the image
        """
        if self.coord_x > self.h - 1 or self.coord_x < 0 or self.coord_y > self.w - 1 or self.coord_y < 0:
            # we then check if the coordinates are equal to (-1, -1) or (-2, -2)
            format1 = (self.coord_x, self.coord_y) == (-1, -1)
            format2 = (self.coord_x, self.coord_y) == (-2, -2)
            if not format1 or not format2:
                return "The head coordinates ({},{}) associated with the image are not relevant".format(
                    self.h, self.w)


if __name__ == "__main__":

    DATA_PATH = Path("../output/test/test_file.csv")
    IMAGE_PATH = Path("../output/test/vid0/f4_c3.jpg")
    data, coords = load_data(DATA_PATH)
    image, coords_h = associate_head_coordinates_with_image(IMAGE_PATH, data)
    list_pairs = augment_data(coords_h, image)

    # display the images to test results
    sess = tf.InteractiveSession()
    for i in range(len(list_pairs)):

        try:
            image = list_pairs[i][0].eval()
            print(list_pairs[i][1])
            image_p = np.int_(image)
            b, g, r = cv2.split(image_p)  # get b,g,r
            image_p = cv2.merge([r, g, b])
            plt.figure()
            plt.imshow(image_p)
            plt.scatter(list_pairs[i][1][0], list_pairs[i][1][1], c='red')
            plt.show()

        except HeadCoordMatchingError as head_coord_matching_error:
            print(head_coord_matching_error.__repr__())
    sess.close()
