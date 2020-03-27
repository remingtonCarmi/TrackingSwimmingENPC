""" to convert the format of a numpy array image"""


def bgr_to_rgb(image):
    """
    To convert a BGR image into a RGB one
    Args:
        image(numpy array): image to convert

    Returns:
        im(numpy array): new image, in RGB
    """
    new_im = image.copy()
    new_im[:, :, 0], new_im[:, :, 2] = image[:, :, 2], image[:, :, 0]
    return new_im