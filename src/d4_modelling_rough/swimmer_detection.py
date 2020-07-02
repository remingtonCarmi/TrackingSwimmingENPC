"""
This code allows to identify a swimmer in an photo with a red spectrum method.
"""
from src.d4_modelling_rough.gradient.gradient import compute_gradient
from src.d4_modelling_rough.skin_colors.skin_colors import keep_skin

import matplotlib.pyplot as plt


plt.rcParams['image.cmap'] = 'gray'


def edges(image, threshold=8, sigma=3, method=2, figures=False):
    """
    From a given image, compute a binary image with swimmer's pixels in white, others in black

    Args:
        image (numpy array): input image
        sigma (integer): parameter for the gradient calculation, see compute_gradient docstring
        threshold (integer): parameter of the method
        method (integer): method of extraction of colors. See load_red docstring
        figures (boolean): if True, plot the returned image

    Returns:
        threshold_gradient (numpy array): binary image with swimmer's pixels in white, others in black
    """
    red_image = keep_skin(image, method) * 255
    gradient = compute_gradient(red_image, sigma)
    threshold_gradient = gradient > threshold

    if figures:
        plt.figure()
        plt.imshow(red_image)
        plt.figure()
        plt.imshow(gradient)
        plt.figure()
        plt.imshow(threshold_gradient)
    return threshold_gradient


if __name__ == "__main__":
    NAME = "..\\..\\tries\\l4_f.jpg"

    IMAGE = plt.imread(NAME)
    plt.figure()
    plt.imshow(IMAGE)

    BINARY = edges(IMAGE, threshold=8, sigma=3, figures=True)
    plt.show()
