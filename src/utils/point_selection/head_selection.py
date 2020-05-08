from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow, QColor
from PyQt5.QtWidgets import QMessageBox, QLayout, QDesktopWidget
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import cv2
from src.utils.point_selection.image_selection.image_selection import ImageSelection
from src.utils.point_selection.main_widget.main_widget import MainWidget


INSTRUCTIONS = "INSTRUCTIONS : \n \n"
INSTRUCTIONS += "   Select a head : left click. \n \n"
INSTRUCTIONS += "   Withdraw a head : press 'w'. \n \n"
INSTRUCTIONS += "   Head not seen : press 'p'. \n \n"
INSTRUCTIONS += "   Valid decision : press space bar. \n \n"
INSTRUCTIONS += "   Zoom in : keep left click, move and release. \n \n"
INSTRUCTIONS += "   Zoom out : click anywhere. \n \n"
INSTRUCTIONS += "   Quit and Save : press 's'."


BLANK = " \n \n \n \n \n \n \n \n \n "


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


def head_selection(image):
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
    image_size = QSize(screen_size.width() - 25, screen_size.height() - 850)

    # Set the image selection and the editable text
    pix_map = array_to_qpixmap(image)
    image_selection = ImageSelection(pix_map, image_size, points_image, colors, skip=True, can_stop=True)

    # Calibration points
    blank = QLabel(BLANK)
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
    ROOT_IMAGE = Path('../../../output/test/raw_images/vid0_frame126.jpg')
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # Select the points
    print(head_selection(IMAGE))
