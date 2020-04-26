"""
The file allows the user to select points in an image,
to tell the real position of these points.
"""

from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize
import cv2
from src.utils.point_selection.information_points.calibration_points import CalibrationPoints
from src.utils.point_selection.image_selection.image_selection import ImageSelection
from src.utils.point_selection.main_widget.main_widget import MainWidget
from src.utils.point_selection.instructions.instructions import instructions_perspective


def array_to_qpixmap(image):
    """
    Transform an array in a QPixmap.

    Args:
        image (array, 2 dimensions, rgb format): the image.

    Returns:
        (QPixmap, brg format): the QPixmap.
    """
    (height, width) = image.shape[:2]
    bytes_per_line = 3 * width
    # If the format is not good : put Format_RGB888
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)

    return QPixmap.fromImage(qimage)


def calibration_selection(image):
    """
    Displays an image on which the user can select points and tell their positions.

    Args:
        image (array, 2 dimensions): the image on which the points will be selected.

    Returns:
        points_image (array, shape = (4, 2)): the select points.

        points_real (array, shape = (4, 2)): the given points.

    Interaction events :
        - if the user click a point is selected
        - if the user click and drag, it zooms
        - if the user press the escape button, it erases the last point
        - if the user press the space bar, the application is closed.
    """
    # Set the points
    colors = [Qt.black, Qt.red, Qt.darkGreen, Qt.darkGray]
    points_image = np.ones((len(colors), 2), dtype=np.float32) * -2
    points_real = np.ones((len(colors), 2), dtype=np.float32) * -2

    # Set application, window and layout
    app = QApplication([])
    instructions_perspective()
    window = MainWidget()
    layout = QHBoxLayout()

    # Get the sizes
    screen_size = QDesktopWidget().screenGeometry()
    screen_ration = 4 / 5
    image_size = QSize(screen_size.width() * screen_ration - 115, screen_size.height() - 150)
    point_size = QSize(screen_size.width() * (1 - screen_ration) - 115, screen_size.height() - 150)

    # Set the image selection and the editable text
    pix_map = array_to_qpixmap(image)
    image_selection = ImageSelection(pix_map, image_size, points_image, colors)
    information_points = CalibrationPoints(point_size, colors, points_real)

    # Add widgets to layout
    layout.addWidget(image_selection)
    layout.addLayout(information_points)

    # Add layout to window and show the window
    window.setLayout(layout)
    window.showMaximized()
    app.exec_()

    return points_image, points_real


if __name__ == "__main__":
    # Get the array
    ROOT_IMAGE = Path('../../../data/images/raw_images/vid0_frame126.jpg')
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # Select the points
    print(calibration_selection(IMAGE))
