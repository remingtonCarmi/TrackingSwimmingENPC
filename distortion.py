"""This code withdraws the distortion of an image."""
import numpy as np
import cv2
from detection import select_points


class SelectionError(Exception):
    """The exception class error to tell that the number of selected points is not correct."""
    def __init__(self, nb_selected_points):
        """
        Args:
            nb_selected_points (integer): the number of selected points.
        """
        self.nb_selected_points = nb_selected_points

    def __repr__(self):
        """"Indicates that the number of selected points is not correct."""
        if self.nb_selected_points < 4:
            begin_message = "The number of selected points is too low"
        else:
            begin_message = "The number of selected points is not a square number"
        end_message = " : {} points.".format(self.nb_selected_points)
        return begin_message + end_message


def find_distortion_charact(image):
    """
    This function finds the characteristics of the distortion.
    The number of selected points should be greater than 4 and
    it should be a square number.

    Args:
        image (array): an image taken by the camera that we want the characteristics.

    Returns:
        (mtx, dist): characteristics of the camera.
    """
    # Selection of the points
    selection = select_points(image)
    nb_selected_points = len(selection)
    nb_points = int(np.sqrt(nb_selected_points))
    if nb_selected_points < 4 or abs(int(nb_points) - np.sqrt(nb_selected_points)) > 0.0001:
        raise SelectionError(nb_selected_points)
    print(selection)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_selected_points, 3), np.float32)
    objp[:, :2] = np.mgrid[0: nb_points, 0: nb_points].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [objp]  # 3d point in real world space

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # We get the right format for imgpoints : 2d points in image plane.
    imgpoints = np.zeros((nb_selected_points, 1, 2), dtype='float32')
    for i in range(nb_selected_points):
        imgpoints[i, 0, 0] = selection[i][0]
        imgpoints[i, 0, 1] = selection[i][1]
    imgpoints = [imgpoints]

    # We get the characteristics
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)[1: 3]


def clear_image(image, characts):
    """
    Withdraws the distortion.py from the image using the characteristics.

    Args:
        image (array): the image that we want to withdraw the distortion.

        characts (list of characteristics): the characteristics of the camera.

    Returns:
        image_clear (array): the image without distortion.
    """
    (h_dim, w_dim) = image.shape[:2]
    (mtx, dist) = characts
    camera_characts = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_dim, h_dim), 1, (w_dim, h_dim))[0]
    # undistort
    image_clear = cv2.undistort(image, mtx, dist, None, camera_characts)

    return image_clear


if __name__ == "__main__":
    try:
        NAME_IMAGE = "test\\test_img.jpg"
        IMAGE = cv2.imread(NAME_IMAGE)
        CHARACT = find_distortion_charact(IMAGE)
        IMAGE2CLEAN = cv2.imread(NAME_IMAGE)
        IMAGE = clear_image(IMAGE2CLEAN, CHARACT)
        cv2.namedWindow("image_clean", cv2.WINDOW_NORMAL)
        cv2.imshow("image_clean", IMAGE)
        cv2.waitKey(0)
    except SelectionError as select_error:
        print(select_error.__repr__())
