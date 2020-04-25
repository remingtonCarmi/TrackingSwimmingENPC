import numpy as np


def draw_rectangle(image, x0, y0, x1, y1, outline=5):
    """
    Turns pixels of an image along a given rectangle into red ones.
    Does not modify the input.
    Args:
        image (numpy array): the input image
        x0 (integer): the x-coordinate of the top left pixel of the rectangle
        y0 (integer): the y-coordinate of the top left pixel of the rectangle
        outline (integer): the thickness, in pixels, of the outline of the rectangle to draw

    Returns:
        image_bis (numpy array): a copy of the input, with the rectangle drawn in red
    """
    image_bis = np.copy(image)
    size_x = abs(x1 - x0)
    size_y = abs(y1 - y0)
    for k in range(size_x):
        for y in range(0, outline):
            image_bis[y0 - y, x0 + k] = [255, 0, 0]
            image_bis[y0 + y + size_y, x0 + k] = [255, 0, 0]
    for k in range(size_y):
        for x in range(-(outline//2), outline//2 + 1):
            image_bis[y0 + k, x0 - x] = [255, 0, 0]
            image_bis[y0 + k, x0 + x + size_x] = [255, 0, 0]

    return image_bis
