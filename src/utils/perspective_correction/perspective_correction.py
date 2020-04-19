import cv2
import numpy as np


def get_perspective_matrix(src, dst):
    # we find the transform matrix thanks to the matching of the four points
    src = np.float32(src)
    dst = np.float32(dst)
    return cv2.getPerspectiveTransform(src, dst)


def correct_perspective_img(image, perspective_matrix):
    # warp the image to a top-down view
    return cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    SRC = np.zeros((4, 2))
    DST = np.ones((4, 2))
    print(get_perspective_matrix(SRC, DST))
