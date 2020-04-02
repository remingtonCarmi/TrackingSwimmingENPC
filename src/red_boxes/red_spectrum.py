"""
This code allows to identify a swimmer on an photo with the red spectrum method.
@author: Victoria Brami, Maxime Brisinger, Theo Vincent
"""

import numpy as np
# this is the key library for manipulating arrays. Use the online ressources! http://www.numpy.org/

import matplotlib.pyplot as plt
# used to read images, display and plot. 

import scipy.ndimage as ndimage
# one of several python libraries for image procession

import os

from src.utils.extract_image import extract_image_video as extract

plt.rcParams['image.cmap'] = 'gray'
# by default, the grayscale images are displayed with the jet colormap: use grayscale instead

THRESHOLD = 0.1
THRESHOLD_2 = 18  # a preciser
FIG_SIZE = (10, 10)


def load_image(name, crop=None):
    """
    Loads image named NAME
    Args :
        :param name : string, name of the file to load
        :param crop : (list of four integers) : to crop the image into a smaller one
    :return:
    """
    if crop is None:
        crop = [1]
    image = plt.imread(name)
    if crop != [1]:
        image = image[crop[0]:crop[1], crop[2]:crop[3]]
    return image


def load_red(image, method=2):
    """
    Uses the 3 components of a pixel to highlight skin colors, returns the image as a numpy.ndarray

    Args:
        image (numpy array)
        method (integer): method of extraction of colors
            method = 1: keep the difference between the red and the blue component
            method = 2: use precises criterias verified by most skin colors to extract the correct pixels
    Returns:
        image (numpy array):
    """

    if method == 1:
        image = 100. * image[:, :, 0] - 99. * image[:, :, 2]
        # image = image.astype('float') / 255  # just to scale the values of the image between 0 and 1

    if method == 2:
        red_image = image[:, :, 0]
        green_image = image[:, :, 1]
        blue_image = image[:, :, 2]
        # image = image[:, :, 0] < -1

        r1 = red_image < 190
        r2 = red_image > 120
        g = green_image < 220
        b = blue_image < 230

        image = ((1*r1 + 1*r2 + 1*g + 1*b) == 4)


        # for i in range(image.shape[0] - 1):
        #     for j in range(image.shape[1] - 1):
        #         if 120 < red_image[i, j] < 190 and green_image[i, j] < 220 and blue_image[i, j] < 230:
        #             image[i, j] = 1
    return image


def compute_gradient(image, sigma=0):
    """
    Computes the vertical, horizontal, and the norm of the gradient of an image image.
    Returns them as three numpy.ndarray

    Args:
        image(numpy.ndarray) : image we want the gradient of
        
        sigma(integer) : value of the parameter of the gaussian filter the algorithm apply
    """
    image = ndimage.gaussian_filter(image, sigma, mode='constant', cval=0)
    y = np.array([[1. / 9, 0, -1. / 9],
                  [2. / 9, 0, -2. / 9],
                  [1. / 9, 0, -1. / 9]])
    x = np.array([[1. / 9, 2. / 9, 1. / 9],
                  [0, 0, 0],
                  [-1. / 9, -2. / 9, -1. / 9]])
    gradient_y = ndimage.convolve(image, y)
    gradient_x = ndimage.convolve(image, x)
    gradient_norm = np.sqrt(gradient_y ** 2 + gradient_x ** 2)
    return gradient_norm


def keep_edges(image, method=2, figures=False):
    """
    Treats an image to only keep the edges of the swimmer

    Args:
        image (string): name of the file
        method (integer): method of extraction of colors
        figures (boolean): if True, display the gradient and the threshold gradient

    Returns:
        threshold_gradient (numpy array): image after threshold the gradient
    """
    red_image = load_red(image, method)

    threshold_image = (red_image > THRESHOLD) * 255

    gradient = compute_gradient(threshold_image, 3)
    threshold_gradient = gradient > THRESHOLD_2

    if figures:
        plt.figure(figsize=FIG_SIZE)
        plt.imshow(gradient)
        plt.figure(figsize=FIG_SIZE)
        plt.imshow(threshold_gradient)
    return threshold_gradient


def extreme_white_pixels(image):
    """
    Among the white pixels of a binary image, finds the top left one and the bottom right one
    
    Args:
        image(numpy array) : binary image
    
    Returns:
        (x_min, y_min), (x_max, y_max) : 2 couples of coordinates
    """

    y_min, x_min = image.shape[0], image.shape[1] # remettre - 1
    # y_max, x_max = 0, 0
    # for y in range(image.shape[0] - 1):
    #     for x in range(image.shape[1] - 1):
    #         if image[y, x]:
    #             if y < y_min:
    #                 y_min = y
    #             if y > y_max:
    #                 y_max = y
    #             if x < x_min:
    #                 x_min = x
    #             if x > x_max:
    #                 x_max = x
    a = np.linspace(0, x_min - 1, x_min)
    b = np.linspace(0, y_min - 1, y_min)
    c, d = np.meshgrid(a, b)
    ci = c * image
    di = d * image
    return [int(np.min(ci + 10000*(ci == 0))), int(np.min(di + 10000*(di == 0)))], [int(np.max(ci)), int(np.max(di))]
    # return [x_min, y_min], [x_max, y_max]


def get_rectangle(gradient, offset=[0, 0]):
    """
    To get characteristics of the boxes surrounding the swimmers
    Args:
        gradient (numpy array): a binary image, with white pixels only for the swimmer
        offset (list of 2 integers): to move the box horizontally and vertically

    Returns:
        extremes[0]: the coordinates of top left corner of the box
        size (list of 2 integers) : the x_size and the y_size of the box
    """
    extremes = extreme_white_pixels(gradient)
    size = (extremes[1][0] - extremes[0][0], extremes[1][1] - extremes[0][1])
    extremes[0][0] += offset[0]
    extremes[0][1] += offset[1]
    return extremes[0], size


def draw_rectangle(corner, size, draw=True):
    """
    To build a given rectangle as a plt object, and plot it on the opened figure
    Args:
        corner (list of 2 integers): the coordinates of top left corner of the box
        size (list of 2 integers) : the x_size and the y_size of the box
        draw (boolean): if True, plot the rectangle on the opened figure

    Returns:
        rectangle (patch de plt):
    """
    rectangle = plt.Rectangle(corner, size[0], size[1], fc="none", ec="red")
    if draw:
        plt.gca().add_patch(rectangle)
    return rectangle


if __name__ == "__main__":
    NAME = "..\\..\\Figure_2.jpg"
    if not os.path.exists(NAME):
        extract("videos\\Florent Manaudou Wins Men's 50m Freestyle Gold -- London 2012 Olympics", 180, 180, True,
                "test\\")

    # # later, it will be calculated automatically
    # CROPS = [[400, 720, 210, 360],
    #          [350, 720, 400, 550],
    #          [400, 800, 600, 735],
    #          [200, 1100, 800, 950],
    #          [400, 900, 990, 1130],
    #          [360, 900, 1200, 1310],
    #          [300, 900, 1380, 1480],
    #          [430, 800, 1580, 1680],
    #          ]

    # for CROP in CROPS:
    #     IMAGE = load_image(NAME, CROP)
    #     plt.figure(figsize=FIG_SIZE)
    #     plt.imshow(IMAGE)
    #     print(np.shape(IMAGE))
    #     CORNER, SIZE = get_rectangle(keep_edges(IMAGE, 2, False))
    #     draw_rectangle(CORNER, SIZE)

    IMAGE = load_image(NAME)
    plt.figure()
    plt.imshow(IMAGE)
    CORNER, SIZE = get_rectangle(keep_edges(IMAGE, 2, True))

    draw_rectangle(CORNER, SIZE)

    plt.show()
