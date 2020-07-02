from scipy import ndimage
import numpy as np


def compute_gradient(image, sigma=0):
    """
    Computes the norm of the gradient of an image.
    Args:
        image(numpy array) : image we want the gradient of
        sigma(integer) : parameter of the gaussian filter the algorithm apply

    Returns:
        gradient_norm (numpy array): the norm of the gradient of the input image
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
