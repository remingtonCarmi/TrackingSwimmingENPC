"""
This module plot a lane and ask the user to spot the head of the swimmer.
"""
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtCore import Qt, QSize
import cv2

# To plot the lane and select the head of the swimmer
from src.d0_utils.point_selection.image_selection.image_selection import ImageSelection

# To get the main widget
from src.d0_utils.point_selection.main_widget.main_widget import MainWidget


# Set the instructions for the selection
INSTRUCTIONS = "INSTRUCTIONS : \n \n"
INSTRUCTIONS += "   Select a head : left click. \n \n"
INSTRUCTIONS += "   Withdraw a head : press 'w'. \n \n"
INSTRUCTIONS += "   Head not seen : press 'p'. \n \n"
INSTRUCTIONS += "   Valid decision : press space bar. \n \n"
INSTRUCTIONS += "   Zoom in : keep left click, move and release. \n \n"
INSTRUCTIONS += "   Zoom out : click anywhere. \n \n"
INSTRUCTIONS += "   Quit and Save : press 's'."

BLANK = " \n \n \n "


def array_to_qpixmap(image):
    """
    Transform an array in a QPixmap.

    Args:
        image (array, 2 dimensions, bgr format): the image.

    Returns:
        (QPixmap, brg format): the QPixmap.
    """
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    # If the format is not good : put Format_BGR888
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(qimage)


def head_selection(image, lane, frame):
    """
    Plot the image and ask to point at the head of the swimmer.

    Args:
        image (array): the image to point at.

        lane (integer): the index of the lane to selection.

        frame (integer): the index of the frame to selection.

    Returns:
        points_image (array of shape : (1, 2)): the coordinates of the selected point.

        (boolean): says if the user wants to stop the pointing.
    """
    # Set the points
    colors = [Qt.black]
    points_image = np.ones((len(colors), 2)) * -2

    # Set application, window and layout
    app = QApplication([])
    # index_close is the index of the child that is the image selection widget
    window = MainWidget(index_close=2)
    layout = QVBoxLayout()

    # Get the sizes
    screen_size = QDesktopWidget().screenGeometry()
    image_size = QSize(screen_size.width() - 25, screen_size.height() - 900)

    # Set the image selection and the editable text
    pix_map = array_to_qpixmap(image)
    image_selection = ImageSelection(pix_map, image_size, points_image, colors, skip=True, can_stop=True)

    # Calibration points
    blank_frame_lane = BLANK + "Lane n° {} \n \n Frame n° {}".format(lane, frame) + BLANK
    blank = QLabel(blank_frame_lane)
    information_points = QLabel(INSTRUCTIONS)

    # Add widgets to layout
    layout.addWidget(blank)
    layout.addWidget(image_selection)
    layout.addWidget(information_points)

    # Add layout to window and show the window
    window.setLayout(layout)
    window.showMaximized()
    app.exec_()

    return points_image, image_selection.stop


if __name__ == "__main__":
    # Get the array
    ROOT_IMAGE = Path('../../../data/4_model_output/tries/raw_images/vid0_frame126.jpg')
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # Select the points
    print(head_selection(IMAGE, 1, 2))
