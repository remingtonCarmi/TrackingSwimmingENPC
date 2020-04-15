from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import cv2


class ImageSelection(QLabel):
    def __init__(self, pix_map, size, points, colors):
        super().__init__()

        # Tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(True)

        # Points management
        self.list_point = np.zeros(20, dtype=QPoint)
        self.points = points
        self.register = False
        self.nb_points = 0
        self.cursorPos = QPoint()
        self.pStart = QPoint()
        self.pEnd = QPoint()

        # Background management
        self.colors = colors
        self.real_image_size = pix_map.size()
        self.pix_map = pix_map.scaled(size, Qt.IgnoreAspectRatio)
        self.setFixedSize(self.pix_map.size())
        top_left = QPoint()
        bottom_right = QPoint(self.size().width() - 1, self.size().height() - 1)
        self.rect_in_image = QRect(top_left, bottom_right)
        self.zoom_in = False
        self.setPixmap(pix_map)

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

    def closeEvent(self, event):
        # closeEvent is called twice since the super needs to be closed
        if not self.register:
            self.register = True
            for index_point in range(self.nb_points):
                x_select = self.list_point[index_point].x()
                y_select = self.list_point[index_point].y()
                x_image = int((x_select / self.size().width()) * self.real_image_size.width())
                y_image = int((y_select / self.size().height()) * self.real_image_size.height())
                self.points.append(np.array([x_image, y_image]))

    def update_points(self):
        # Add a point if the mouse did not move
        if self.pStart == self.pEnd:
            # There should be has much as selected points as colors
            if self.nb_points >= len(self.colors):
                super().close()
                self.close()
            # Withdraw the zoom and add the point
            if self.zoom_in:
                # Withdraw the zoom
                self.setPixmap(self.pix_map.scaled(self.size(), Qt.IgnoreAspectRatio))
                self.zoom_in = False
                # Add the point
                self.list_point[self.nb_points] = self.point_in_image()
                # Reset the rectangle
                top_left = QPoint()
                bottom_right = QPoint(self.size().width() - 1, self.size().height() - 1)
                self.rect_in_image = QRect(top_left, bottom_right)
            else:
                self.list_point[self.nb_points] = self.pStart
            self.nb_points += 1
        # Zoom if the mouse mouved
        else:
            self.zoom()
            self.zoom_in = True

    def draw_points(self, q_painter):
        # Draw a circle that follows the mouse
        q_painter.setPen(Qt.white)
        if not self.cursorPos.isNull():
            q_painter.drawEllipse(self.cursorPos.x() - 2, self.cursorPos.y() - 2, 4, 4)

        # Draw the points
        for index_point in range(self.nb_points):
            q_painter.setPen(self.colors[index_point])
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            if self.in_showed_image(x, y):
                (x, y) = self.adapt_point(x, y)
                q_painter.drawEllipse(x - 2, y - 2, 4, 4)

    def zoom(self):
        rect_in_screen = self.rectangle_in_screen()
        self.rect_in_image = self.rectangle_in_image(rect_in_screen)

        zoom_pix = self.pix_map.copy(self.rect_in_image)
        self.setPixmap(zoom_pix.scaled(self.size(), Qt.IgnoreAspectRatio))

    def rectangle_in_screen(self):
        top_left_x = max(min(self.pStart.x(), self.pEnd.x()), 0)
        top_left_y = max(min(self.pStart.y(), self.pEnd.y()), 0)
        top_left = QPoint(top_left_x, top_left_y)

        bottom_right_x = min(max(self.pStart.x(), self.pEnd.x()), self.size().width())
        bottom_right_y = min(max(self.pStart.y(), self.pEnd.y()), self.size().height())
        bottom_right = QPoint(bottom_right_x, bottom_right_y)

        return QRect(top_left, bottom_right)

    def rectangle_in_image(self, rectangle_zoom):
        start_x = self.rect_in_image.topLeft().x()
        start_y = self.rect_in_image.topLeft().y()

        top_left_x = start_x + (rectangle_zoom.topLeft().x() / self.size().width()) * self.rect_in_image.width()
        top_left_y = start_y + (rectangle_zoom.topLeft().y() / self.size().height()) * self.rect_in_image.height()
        top_left = QPoint(int(top_left_x), int(top_left_y))

        bottom_right_x = start_x + (rectangle_zoom.bottomRight().x() / self.size().width()) * self.rect_in_image.width()
        bottom_right_y = start_y + (rectangle_zoom.bottomRight().y() / self.size().height()) * self.rect_in_image.height()
        bottom_right = QPoint(int(bottom_right_x), int(bottom_right_y))

        return QRect(top_left, bottom_right)

    def point_in_image(self):
        start_x = self.rect_in_image.topLeft().x()
        start_y = self.rect_in_image.topLeft().y()

        point_x = start_x + (self.pStart.x() / self.size().width()) * self.rect_in_image.width()
        point_y = start_y + (self.pStart.y() / self.size().height()) * self.rect_in_image.height()

        return QPoint(int(point_x), int(point_y))

    def in_showed_image(self, x_coord, y_coord):
        if self.rect_in_image.left() < x_coord < self.rect_in_image.right():
            if self.rect_in_image.top() < y_coord < self.rect_in_image.bottom():
                return True
        return False

    def adapt_point(self, x_coord, y_coord):
        in_zoom_x = (x_coord - self.rect_in_image.topLeft().x()) * self.size().width() / self.rect_in_image.width()
        in_zoom_y = (y_coord - self.rect_in_image.topLeft().y()) * self.size().height() / self.rect_in_image.height()

        return in_zoom_x, in_zoom_y


"""def initialize_points(size):
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

    main_layout.addLayout(grid)"""


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

    SCREE_RATIO = 4 / 5

    POINTS = []
    COLORS = [Qt.black, Qt.red, Qt.yellow, Qt.gray]
    # Set application, window and layout
    app = QApplication([])
    window = QWidget()
    window.show()
    layout = QHBoxLayout()

    # Get the sizes
    screen_size = QDesktopWidget().screenGeometry()
    image_size = QSize(screen_size.width() * SCREE_RATIO - 115, screen_size.height() - 150)
    point_size = QSize(screen_size.width() * (1 - SCREE_RATIO) - 115, screen_size.height() - 150)

    # Set image selection zone
    pix_map = array_to_qpixmap(IMAGE)
    image_selection = ImageSelection(pix_map, image_size, POINTS, COLORS)
    # information_points = initialize_points(point_size)

    # Add widgets to layout
    layout.addWidget(image_selection)
    # layout.addWidget(information_points)

    # Add layout to window and show the window
    window.setLayout(layout)
    window.showMaximized()
    app.exec_()
    print("Points", POINTS)

