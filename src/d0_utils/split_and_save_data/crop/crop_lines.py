"""
This code separates an image in several images, delimited by given lines.
"""


def crop(image, list_y, margin=0):
    """
    To vertically crop an images at given positions.
    Args:
        margin (integer): number of lines of pixels
            - we remove (if margin > 0)
            - we add (if margin < 0)

        image(numpy array): image to crop.

        list_y(list of integers): list of vertical positions where we crop.

    Returns:
        images_crop(list of numpy arrays): list of cropped images.
    """

    images_crop = []

    y_prev = list_y[0]

    if margin >= 0:
        for y in list_y[1:]:
            images_crop.append(image[y_prev + margin: y - margin, :])
            y_prev = y
    else:
        y = list_y[1]
        images_crop.append(image[y_prev: y - 2 * margin, :])  # * 2 so that the images have the same size
        y_prev = y

        for y in list_y[2:-1]:
            images_crop.append(image[y_prev + margin: y - margin, :])
            y_prev = y

        y = list_y[-1]
        images_crop.append(image[y_prev + 2 * margin: y, :])  # * 2 so that the images have the same size

    return images_crop


def crop_list(list_images, points, margin=0):
    """
    Crop a list of images.

    Args:
        list_images (list of array): list of the images.

        points (list of integers): list of vertical positions where we crop.

        margin (integer): number of lines of pixels.
            - we remove (if margin > 0)
            - we add (if margin < 0)

    Returns:
        list_images_crop (list of array): list of the cropped images.
    """
    list_images_crop = []

    for image in list_images:
        list_images_crop.append(crop(image, points, margin))

    return list_images_crop
