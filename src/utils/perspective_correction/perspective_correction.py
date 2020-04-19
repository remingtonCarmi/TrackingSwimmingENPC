import cv2


def get_perspective_matrix(src, dst):
    # we find the transform matrix thanks to the matching of the four points
    return cv2.getPerspectiveTransform(src, dst)


def correct_perspective_img(image, perspective_matrix):
    # warp the image to a top-down view
    return cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


if __name__ == "__name__":
    print("OK")
