import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QPoint
import random


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.list_point = []
        self.nb_points = 0
        self.cursorPos = QPoint(0, 0)
        self.pStart = QPoint(0, 0)
        self.pEnd = QPoint(0, 0)
        self.rectangle1 = QPoint(0, 0)
        self.rectangle2 = QPoint(0, 0)

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
        print("press: ", self.pStart)

    def mouseReleaseEvent(self, event):  # evenement mouseRelease
        self.pEnd = event.pos()
        print("release: ", event.pos())
        self.update_points()

    def update_points(self):
        if not self.pStart.isNull():
            if self.pStart == self.pEnd:
                self.list_point.append(self.pStart)
                self.nb_points += 1
            else:
                self.rectangle1 = self.pStart
                self.rectangle2 = self.pEnd

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

        if not self.rectangle1.isNull():
            x_1 = self.rectangle1.x()
            y_1 = self.rectangle1.y()
            x_2 = self.rectangle2.x()
            y_2 = self.rectangle2.y()
            q_painter.drawRect(min(x_1, x_2), min(y_1, y_2), abs(x_2 - x_1), abs(y_2 - y_1))

    def get_points(self):
        points = np.zeros((self.nb_points, 2))

        for index_point in range(self.nb_points):
            x = self.list_point[index_point].x()
            y = self.list_point[index_point].y()
            points[index_point, :] = np.array([x, y])

        return points


if __name__ == "__main__":
    # Set application, window and layouts
    app = QApplication([])
    window = QMainWindow()
    window.setCentralWidget(Example())
    window.resize(window.maximumWidth(), window.maximumHeight())
    window.show()
    app.exec_()

