import numpy as np


def extreme_white_pixels(image):
    """
    Among the white pixels of a binary image, finds the minimum and maximum x and y

    Args:
        image(numpy array) : a binary image

    Returns:
        x_min, y_min, x_max, y_max (integers): 2 couples of coordinates, in pixels
    """

    y_min, x_min = image.shape[0], image.shape[1]

    a = np.linspace(0, x_min - 1, x_min)
    b = np.linspace(0, y_min - 1, y_min)
    c, d = np.meshgrid(a, b)
    ci = c * image
    di = d * image

    return (int(np.min(ci + 2 * x_min * (ci == 0))),
            int(np.min(di + 2 * y_min * (di == 0))),
            int(np.max(ci)),
            int(np.max(di))
            )
