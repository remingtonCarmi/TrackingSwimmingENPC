import numpy as np
import cv2
import glob
from detection import select_points


def find_distortion_charact(image_path, nb_points_height=2, nb_points_width=2):
    """
    This function finds the characteristics of the distortion.py.
    
    Args:
        image_path
        
        nb_points_height
        
        nb_points_width
    
    Returns:
        charac_dist (mtx, dist)
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_points_width * nb_points_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0: nb_points_height, 0: nb_points_width].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [objp]  # 3d point in real world space

    images = glob.glob(image_path)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Selection of the points
    selection = select_points(image_path, nb_points_width * nb_points_height)
    print(selection)

    # We get the right format for imgpoints : 2d points in image plane.
    imgpoints = np.zeros((nb_points_height * nb_points_width, 1, 2), dtype='float32')
    for i in range(nb_points_height * nb_points_width):
        imgpoints[i, 0, 0] = selection[i][0]
        imgpoints[i, 0, 1] = selection[i][1]
    imgpoints = [imgpoints]

    # We get the characteristics
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)[1: 3]


def clear_image(image, characts):
    """
    Withdraws the distortion.py from the image using the characterics.

    Args:
        image_path

        characts

    Returns:
        image_clear
    """
    (h, w) = image.shape[:2]
    (mtx, dist) = characts
    camera_characts = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))[0]
    # undistort
    image_clear = cv2.undistort(image, mtx, dist, None, camera_characts)

    return image_clear


if __name__ == "__main__":
    charact = find_distortion_charact("test\\frame2.jpg")
    image2clean = cv2.imread("test\\frame2.jpg")
    image = clear_image(image2clean, charact)
    cv2.namedWindow("image_clean", cv2.WINDOW_NORMAL)
    cv2.imshow("image_clean", image)
    cv2.waitKey(0)
