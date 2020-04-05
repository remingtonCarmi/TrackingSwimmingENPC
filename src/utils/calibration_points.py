import functools
import numpy as np
from src.utils.point_selection import select_points
from pathlib import Path
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QMessageBox, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QGridLayout


class MessageAndPoints:
    def __init__(self):
        self.width_line = 2.5
        self.nb_points = 1
        self.nb_lines = 11
        self.lengths = [0, 5, 15, 35, 45, 50]
        self.nb_lengths = len(self.lengths)
        self.points = np.zeros((self.nb_lines, self.nb_lengths, 2, 3))

    def update_points(self, points, index_line, index_length):
        for (point2d_x, poind2d_y) in points:
            point3d_x = self.width_line * index_line
            point3d_y = self.lengths[index_length]
            new_point = np.array([[point2d_x, poind2d_y, 1], [point3d_x, point3d_y, 0]])
            self.points[index_line, index_length] = new_point

    def update_button(self, button, index_line, index_length):
        point = self.points[index_line, index_length]
        point_2d = "{},".format(point[0, 0: 2])
        point_3d = " {};".format(point[1])
        button.setText(point_2d + point_3d)


def initialize_grid(grid_points, manager, image):
    nb_lengths = manager.nb_lengths
    lengths = manager.lengths
    nb_lines = manager.nb_lines

    for idx_line in range(nb_lines):
        for idx_length in range(nb_lengths):
            push_but = QPushButton("[0. 0.], [0. 0. 0.];")
            grid_points.addWidget(push_but, idx_line + 1, idx_length + 1)
            push_but.clicked.connect(functools.partial(one_selection, image, manager, push_but, idx_line, idx_length))

    for idx_length in range(1, nb_lengths + 1):
        grid_points.addWidget(QLabel("Length {} meters".format(lengths[idx_length - 1])), 0, idx_length)

    for idx_line in range(1, nb_lines + 1):
        grid_points.addWidget(QLabel("Line {}".format(idx_line - 1)), idx_line, 0)


def one_selection(image, manager, button, index_line, index_length):
    # Update points
    selected_points = select_points(image, manager.nb_points)
    manager.update_points(selected_points, index_line, index_length)
    # Update button
    manager.update_button(button, index_line, index_length)


def define_points(image):
    message_points = MessageAndPoints()

    # Set application, window and layouts
    app = QApplication([])
    window = QWidget()
    layout_button = QHBoxLayout()
    layout = QVBoxLayout()
    grid = QGridLayout()

    # Set buttons and labels
    initialize_grid(grid, message_points, image)

    # Set connections
    layout.addLayout(layout_button)
    layout.addLayout(grid)

    # Show the buttons
    window.setLayout(layout)
    window.show()
    app.exec_()


if __name__ == "__main__":
    # print("Click left to select the point, press w to withdraw the last point, press q to exit.")
    IMAGE_PATH = Path("../../output/test/vid0_frame2.jpg")
    IMAGE = cv2.imread(str(IMAGE_PATH))
    define_points(IMAGE)
