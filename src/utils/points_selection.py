from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout
from PyQt5.QtGui import QPainter, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRectF
import cv2


class ImageSelection(QWidget):

    def __init__(self, list_points):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(True)
        self.final_list_points = list_points
        self.list_point = np.zeros(20, dtype=QPoint)
        self.nb_points = 0
        self.cursorPos = QPoint(0, 0)
        self.pStart = QPoint(0, 0)
        self.pEnd = QPoint(0, 0)
        self.top_left = QPoint(0, 0)
        self.bottom_right = QPoint(0, 0)
        self.rectangle = QRectF(self.top_left, self.bottom_right)
        self.finish = False

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        self.draw_points(painter)
        painter.end()

    def mouseMoveEvent(self, event):  # evenement mouseMove
        self.cursorPos = event.pos()  # on stocke la position du curseur
        self.update()

    def mousePressEvent(self, event):  # evenement mousePress
        self.pStart = event.pos()

    def mouseReleaseEvent(self, event):  # evenement mouseRelease
        self.pEnd = event.pos()
        self.update_points()
        self.update()

    def keyReleaseEvent(self, event):
        if self.nb_points > 0 and event.key() == Qt.Key_Control:
            self.nb_points -= 1
        if event.key() == Qt.Key_Space:
            self.close()
        self.update()

    def closeEvent(self, event):
        self.final_list_points = self.list_point

    def update_points(self):
        if not self.pStart.isNull():
            if self.pStart == self.pEnd:
                self.list_point[self.nb_points] = self.pStart
                self.nb_points += 1
            else:
                self.top_left.setX(min(self.pStart.x(), self.pEnd.x()))
                self.top_left.setY(min(self.pStart.y(), self.pEnd.y()))
                self.bottom_right.setX(max(self.pStart.x(), self.pEnd.x()))
                self.bottom_right.setY(max(self.pStart.y(), self.pEnd.y()))
                self.rectangle = QRectF(self.top_left, self.bottom_right)

    def draw_points(self, q_painter):
        q_painter.setPen(Qt.black)

        if not self.cursorPos.isNull():
            q_painter.drawEllipse(self.cursorPos.x() - 2, self.cursorPos.y() - 2, 4, 4)

        q_painter.setPen(Qt.red)

        for index_point in range(self.nb_points):
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            q_painter.drawEllipse(x - 2, y - 2, 4, 4)

        q_painter.setPen(Qt.blue)

        if not self.rectangle.isNull():
            q_painter.drawRect(self.rectangle)

    def get_points(self):
        points = np.zeros((self.nb_points, 2))

        for index_point in range(self.nb_points):
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            points[index_point, :] = np.array([x, y])

        return points


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()

    def mouseMoveEvent(self, event):  # evenement mouseMove
        self.update()
        print("vv")

    def keyReleaseEvent(self, event):
        print("aa")
        if event.key() == Qt.Key_Space:
            self.close()
        self.update()


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
    LIST_POINTS = np.array([1])
    ROOT_IMAGE = Path('../../data/images/raw_images/vid0_frame126.jpg')
    # print(os.listdir(ROOT_IMAGE))
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # Set application, window and layouts
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()

    # Set image selection
    pix_map = array_to_qpixmap(IMAGE)
    label = QLabel()
    label.setPixmap(pix_map)
    label.show()
    image_selection = ImageSelection(LIST_POINTS)

    # Add widget to layout
    layout.addWidget(image_selection)
    initialize_points(layout, window.height())

    # Add layout to window
    window.setLayout(layout)
    window.showMaximized()
    # window.resize(window.maximumWidth(), window.maximumHeight())
    window.show()
    app.exec_()

    # pixmap = QPixmap('../../data/images/raw_images/100NL_FAF.mov_frame126.jpg')
    # window.setPixmap(pixmap)
