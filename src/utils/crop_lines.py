"""
This code separates an image in several images, delimited by given lines
"""


def crop(image, list_y, margin=0):
    """
    To vertically crop an images at given positions
    Args:
        margin (integer):
        image(numpy array): image to crop
        list_y(list of integers): list of vertical positions where we crop

    Returns:
        images_crop(list of numpy arrays): list of cropped images
    """

    images_crop = []

    y_prev = list_y[0]

    for y in list_y[1:]:
        images_crop.append(image[y_prev + margin: y - margin, :])
        y_prev = y
    return images_crop


def crop_list(list_images, points, margin=0):

    list_images_crop = []

    for image in list_images:
        list_images_crop.append(crop(image, points, margin))

    return list_images_crop
