""" This code aims to generate more images from one to increase data"""

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy as sc
from scipy import ndimage
import random as rd
from src.load_data import load_data
from src.head_coords_image_association import associate_head_coordinates_with_image


class HeadCoordMatchingError(Exception):
    """ Verify if the head coordinates are relevant when we look at the corresponding image"""

    def __init__(self, w, h, coord):
        """
        Constructs the image's name.
        """
        self.h = h
        self.w = w
        self.coord_x = coord[0]
        self.coord_y = coord[1]

    def __repr__(self):
        """
        :return: (string) Error message if the coords don't match with the image
        """

        message = "The head coordinates ({},{}) associated with the image of size {}x{} are not relevant".format(
            self.coord_x, self.coord_y, self.h, self.w)
        print(message)


def apply_convolution(img, kernel):
    new_img = img.copy()
    new_img[:, :, 0] = sc.ndimage.convolve(img[:, :, 0], kernel)
    new_img[:, :, 1] = sc.ndimage.convolve(img[:, :, 1], kernel)
    new_img[:, :, 2] = sc.ndimage.convolve(img[:, :, 2], kernel)
    return new_img


def apply_salt_and_pepper(img, sigma):
    new_img = img.copy()
    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        for j in range(w):
            for c in range(3):
                pixel = int(img[i, j, c] + rd.randint(0, sigma))
                if pixel > 255:
                    new_img[i, j, c] = 255
                else:
                    new_img[i, j, c] = pixel
    return new_img


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

    # convolution and blurred images parameters
    # salt and pepper
    sigma = 3

    # blurring this image
    kernel1 = 1 / 25 * np.ones((5, 5))

    # Gaussian 5x5 filter
    kernel3 = 1 / 256 * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]
                                  ])
    kernel4 = 1 / 81 * np.ones((9, 9))

    img1 = apply_convolution(image, kernel1)
    img2 = apply_salt_and_pepper(image, sigma)
    img3 = apply_convolution(image, kernel3)
    img4 = apply_convolution(image, kernel4)
    img1, coord = convert_into_image_tensor(img1, coord)
    img2, coord = convert_into_image_tensor(img2, coord)
    img3, coord = convert_into_image_tensor(img3, coord)
    img4, coord = convert_into_image_tensor(img4, coord)

    img, coord = convert_into_image_tensor(image, coord)
    img = tf.image.convert_image_dtype(img, tf.float32)
    image1 = tf.image.flip_left_right(img)
    image2 = tf.image.random_saturation(img, 0, 3)
    image3 = tf.image.random_brightness(img, max_delta=0.4)
    image4 = tf.image.random_saturation(img, 1, 2)
    image5 = tf.image.random_brightness(img, max_delta=0.2)
    image6 = tf.image.random_contrast(img, 0, 1)
    image7 = tf.image.convert_image_dtype(img1, tf.float32)
    image8 = tf.image.convert_image_dtype(img2, tf.float32)
    image9 = tf.image.convert_image_dtype(img3, tf.float32)
    image10 = tf.image.convert_image_dtype(img4, tf.float32)
    pairs = [[img, coord], [image1, (w - coord[0], h - coord[1])], [image2, coord],
             [image3, coord], [image4, coord], [image5, coord],
             [image6, coord], [image7, coord], [image8, coord],
             [image9, coord], [image10, coord]]

    for i in range(len(pairs)):
        sess = tf.InteractiveSession()
        im = pairs[i][0].eval()
        im = np.int_(im)
        h, w = [im.shape[0], im.shape[1]]
        head_x, head_y = pairs[i][1]
        if head_x > w - 1 or head_x < 0 or head_y > h - 1 or head_y < 0:
            format1 = ((head_x, head_y) == (-1, -1))
            format2 = ((head_x, head_y) == (-2, -2))
            if not format1 or not format2:
                print("image", i)
                raise HeadCoordMatchingError(h, w, [head_x, head_y])
        sess.close()
    return pairs


if __name__ == "__main__":

    DATA_PATH = Path("../output/test/test_file.csv")
    IMAGE_PATH = Path("../output/test/vid0/f4_c3.jpg")
    data, coords = load_data(DATA_PATH)

    image, coords_h = associate_head_coordinates_with_image(IMAGE_PATH, data)
    list_pairs = augment_data(coords_h, image)

    # display the images to test results
    sess = tf.InteractiveSession()
    for i in range(len(list_pairs)):
        image = list_pairs[i][0].eval()
        print(i)
        image_p = np.int_(image)
        b, g, r = cv2.split(image_p)  # get b,g,r
        image_p = cv2.merge([r, g, b])
        plt.figure()
        plt.imshow(image_p)
        plt.scatter(list_pairs[i][1][0], list_pairs[i][1][1], c='red')
        plt.show()

    sess.close()
