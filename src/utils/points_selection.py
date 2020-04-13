from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QPoint, QRect
import cv2


class ImageSelection(QLabel):
    def __init__(self, list_points, pix_map, size):
        super().__init__()

        # Tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(True)

        # Points management
        self.final_list_points = list_points
        self.list_point = np.zeros(20, dtype=QPoint)
        self.nb_points = 0
        self.cursorPos = QPoint(0, 0)
        self.pStart = QPoint(0, 0)
        self.pEnd = QPoint(0, 0)

        # Background management
        self.pix_map = pix_map
        self.size = size
        self.setFixedSize(size)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        self.draw_points(painter)
        painter.end()

    def mouseMoveEvent(self, event):
        self.cursorPos = event.pos()
        self.update()

    def mousePressEvent(self, event):
        self.pStart = event.pos()

    def mouseReleaseEvent(self, event):
        self.pEnd = event.pos()
        self.update_points()
        self.update()

    def keyReleaseEvent(self, event):
        # If control is pressed and a point has been selected
        if self.nb_points > 0 and event.key() == Qt.Key_Control:
            self.nb_points -= 1
        # If space is pressed
        if event.key() == Qt.Key_Space:
            super().close()
            self.close()
        self.update()

    def update_points(self):
        # Add a point if the mouse did not move
        if self.pStart == self.pEnd:
            self.list_point[self.nb_points] = self.pStart
            self.nb_points += 1
        # Zoom if the mouse mouved
        else:
            self.zoom()

    def draw_points(self, q_painter):
        # Draw a circle that follows the mouse
        q_painter.setPen(Qt.black)
        if not self.cursorPos.isNull():
            q_painter.drawEllipse(self.cursorPos.x() - 2, self.cursorPos.y() - 2, 4, 4)

        # Draw the points
        q_painter.setPen(Qt.red)
        for index_point in range(self.nb_points):
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            q_painter.drawEllipse(x - 2, y - 2, 4, 4)

    def get_points(self):
        points = np.zeros((self.nb_points, 2))

        for index_point in range(self.nb_points):
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            points[index_point, :] = np.array([x, y])

        return points

    def zoom(self):
        # Compute the rectangle for the zoom
        top_left = QPoint(0, 0)
        bottom_right = QPoint(0, 0)
        top_left.setX(min(self.pStart.x(), self.pEnd.x()))
        top_left.setY(min(self.pStart.y(), self.pEnd.y()))
        bottom_right.setX(max(self.pStart.x(), self.pEnd.x()))
        bottom_right.setY(max(self.pStart.y(), self.pEnd.y()))
        rectangle = QRect(top_left, bottom_right)
        print(rectangle)
        zoom_pix = self.pix_map.copy(rectangle)
        self.setPixmap(zoom_pix.scaled(self.size, Qt.IgnoreAspectRatio))


def initialize_points(main_layout, height):
    grid = QGridLayout()
    color = ["Blue", "Red", "Yellow", "Green"]

    for index_line in range(4):
        point = QLabel("{} Point".format(color[index_line]))
        text = QTextEdit()

        point.setAlignment(Qt.AlignCenter)
        point.setFixedHeight(height / 4)
        text.setFixedHeight(height / 4)

        grid.addWidget(point, 0, index_line)
        grid.addWidget(text, 1, index_line)

    main_layout.addLayout(grid)


def array_to_qpixmap(image):
    height, width, channel = image.shape
    bytes_per_line = 3 * width

    # If the format is not good : put Format_RGB888
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)

    return QPixmap.fromImage(qimage)


if __name__ == "__main__":
    # Get the array
    LIST_POINTS = np.array([1])
    ROOT_IMAGE = Path('../../data/images/raw_images/vid0_frame126.jpg')
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # Set application, window and layout
    app = QApplication([])
    window = QWidget()
    layout = QHBoxLayout()

    # Set image selection zone
    pix_map = array_to_qpixmap(IMAGE)
    screen_size = QDesktopWidget().screenGeometry().size()
    image_selection = ImageSelection(LIST_POINTS, pix_map, screen_size)
    image_selection.setPixmap(pix_map.scaled(screen_size, Qt.IgnoreAspectRatio))

    # Add widgets to layout
    layout.addWidget(image_selection)
    # initialize_points(layout, window.height())

    # Add layout to window and show the window
    window.setLayout(layout)
    window.showMaximized()
    app.exec_()

