"""
This module computes the homography to apply to an image to get a top-down view
and applies it to an image.
"""
import cv2
import numpy as np


def get_homography(src, dst):
    """
    Compute the homography to get a top-down view

    Args:
        src (array of shape : (4, 2)): the points in pixel in the original image

        dst (array of shape : (4, 2)): the point in meter in the real world

    Returns:
        (array of shape : (3, 3)): the homography matrix
    """
    # we find the transform_image matrix thanks to the matching of the four points
    src = np.float32(src)
    dst = np.float32(dst)
    return cv2.getPerspectiveTransform(src, dst)


def get_top_down_image(image, perspective_matrix):
    """
    Compute the top-down view

    Args:
        image (array): the original image

        perspective_matrix (array of shape : (3, 3)) : the matrix of the homography to apply

    Returns:
        (array): the image with a top-down view
    """
    # warp the image to a top-down view
    return cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    SRC = np.array([[66, 284], [1268, 118], [593, 786], [1888, 391]])
    DST = np.array([[439, 0], [1865, 0], [439, 1081], [1865, 864]])
    print(get_homography(SRC, DST))
