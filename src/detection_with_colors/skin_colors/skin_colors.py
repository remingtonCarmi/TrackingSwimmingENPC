def keep_skin(image, method=2):
    """
    Uses the 3 components of a pixel to highlight skin colors.
    Only keep pixels in a particular range of colors.

    Args:
        image (numpy array): input image
        method (integer): method of extraction of colors
            method = 1: keep the difference between the red and the blue component
            method = 2: use precises criterion verified by most skin colors to extract the correct pixels
    Returns:
        image (numpy array): binary image. Potential skin pixels are in white.
    """

    if method == 1:
        image = 100. * image[:, :, 0] - 99. * image[:, :, 2]

    if method == 2:
        red_image = image[:, :, 0]
        green_image = image[:, :, 1]
        blue_image = image[:, :, 2]

        r1 = red_image < 190
        r2 = red_image > 120
        g = green_image < 220
        b = blue_image < 230

        image = ((1 * r1 + 1 * r2 + 1 * g + 1 * b) == 4)

    return image
