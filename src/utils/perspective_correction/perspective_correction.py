import cv2


def correct_perspective_img(image, src, dst):
    # we find the transform matrix M thanks to the matching of the four points
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    # warp the image to a top-down view
    warped = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped


if __name__ == "__name__":
    print("OK")
