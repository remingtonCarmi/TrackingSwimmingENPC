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

plt.rcParams['image.cmap'] = 'gray' 
# by default, the grayscale images are displayed with the jet colormap: use grayscale instead

NAME = "test//frame1.jpg"
THRESHOLD = 0.1
THRESHOLD_2 = 18
FIG_SIZE = (10,10)

def load_image_red(name, method=2):
    """
    Loads image named NAME and returns its red component as a numpy.ndarray

    Args:
        :param name : string, name of the file to load
        :param method: integer
    """
    image = plt.imread(name)
    if method == 1:
        image = 100.*image[:,:,0] - 99.*image[:,:,2]
        image = image.astype('float')/255 # just to scale the values of the image between 0 and 1 (instead of 0 255)
    if method == 2:
        red_image = image[:,:,0]
        blue_image = image[:,:,2]
        image = image[:,:,0] < -1

        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] -1):
                if 100 < red_image[i, j] < 190 and blue_image[i, j] < 200:
                    image[i,j] = 1

    return image

def compute_gradient(image, sigma=0):
    """
    Computes the vertical, horizontal, and the norm of the gradient of an image image.
    Returns them as three numpy.ndarray

    Args:
        image(numpy.ndarray) : image we want the gradient of
        
        sigma(integer) : value of the parameter of the gaussian filter the algorithm apply
    """
    image = ndimage.gaussian_filter(image, sigma, mode = 'constant', cval = 0)
    y = np.array([[1./9, 0, -1./9],
                  [2./9, 0, -2./9],
                  [1./9, 0, -1./9]])
    x = np.array([[1./9, 2./9, 1./9],
                  [0, 0, 0],
                  [-1./9, -2./9, -1./9]])
    gradient_y = ndimage.convolve(image, y)
    gradient_x = ndimage.convolve(image, x)
    gradient_norm = np.sqrt(gradient_y**2 + gradient_x**2)
    return gradient_y, gradient_x, gradient_norm

def extreme_white_pixels(image):
    image = image > THRESHOLD_2
    y_min, x_min = image.shape[0] - 1, image.shape[1] - 1
    y_max, x_max = 0, 0
    for y in range(image.shape[0] - 1):
        for x in range(image.shape[1] -1):
            if image[y,x]:
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
    return (x_min, y_min),  (x_max, y_max)

def keep_edges(name, method=2, figures=False):
    image = load_image_red(name, method)
    threshold_image = (image > THRESHOLD) * 255
    gradient_y, gradient_x, gradient_norm = compute_gradient(threshold_image, 3)

    if figures:
        plt.figure(figsize=FIG_SIZE)
        plt.imshow(I)
        plt.figure(figsize=FIG_SIZE)
        plt.imshow(I_gradnorm)
        plt.show()
    return gradient_y, gradient_x, gradient_norm

def draw_rectangle(gradient_norm):
    # Draw the rectangle
    extremes = extreme_white_pixels(gradient_norm)
    size = (extremes[1][0] - extremes[0][0], extremes[1][1] - extremes[0][1])

    rectangle = plt.Rectangle(extremes[0], size[0], size[1], fc="none", ec="red")
    plt.gca().add_patch(rectangle)

if __name__ == "__main__":
    # Plot the result
    IMAGE = plt.imread(NAME)
    plt.figure(figsize=FIG_SIZE)
    plt.imshow(IMAGE)
    draw_rectangle(keep_edges(NAME)[2])
    plt.show()