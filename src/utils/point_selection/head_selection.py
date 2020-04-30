from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow, QColor
from PyQt5.QtWidgets import QMessageBox, QLayout, QDesktopWidget
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import cv2
from src.utils.point_selection.information_points.calibration_points import CalibrationPoints
from src.utils.point_selection.image_selection.image_selection import ImageSelection
from src.utils.point_selection.instructions.instructions import instructions_calibration
from src.utils.point_selection.main_widget.main_widget import MainWidget


def array_to_qpixmap(image):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    # If the format is not good : put Format_RGB888
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)

    return QPixmap.fromImage(qimage)


def head_selection(image):
    # Set the points
    colors = [Qt.black]
    points_image = np.ones((len(colors), 2)) * -2

    # Set application, window and layout
    app = QApplication([])
    window = MainWidget()
    layout = QHBoxLayout()

    # Get the sizes
    screen_size = QDesktopWidget().screenGeometry()
    image_size = QSize(screen_size.width() - 25, screen_size.height() - 850)

    # Set the image selection and the editable text
    pix_map = array_to_qpixmap(image)
    image_selection = ImageSelection(pix_map, image_size, points_image, colors, skip=True, can_stop=True)

    # Add widgets to layout
    layout.addWidget(image_selection)

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
