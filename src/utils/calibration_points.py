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

        self.finished = False
        self.empty = True
        self.nb_lines = 11
        self.lengths = [0, 5, 15, 35, 45, 50]
        self.nb_lengths = len(self.lengths)
        self.points = np.zeros((self.nb_lines, self.nb_lengths, 2, 3))

        self.current_nb_line = 0
        self.current_index_length = 0

    def get_message(self):
        if self.finished:
            begin_message = ""
            end_message = "The points selection is finished, please close the window"
        else:
            begin_message = "Select the point on line {} ".format(self.current_nb_line)
            end_message = "at length {} meters.".format(self.lengths[self.current_index_length])

        return begin_message + end_message

    def next(self):
        self.current_nb_line += 1

        if self.current_nb_line == self.nb_lines:
            self.current_nb_line = 0
            self.current_index_length += 1

        if self.current_index_length == self.nb_lengths:
            self.finished = True

    def next_length(self):
        if not self.finished:
            self.current_index_length += 1

            if self.current_index_length == self.nb_lengths:
                self.current_nb_line = self.nb_lines
                self.finished = True

        return self.get_message()

    def back(self):
        if self.current_index_length >= self.nb_lengths:
            self.finished = False
            self.current_index_length = self.nb_lengths - 1
            self.current_nb_line = self.nb_lines - 1

        else:
            self.current_nb_line -= 1

            if self.current_nb_line < 0:
                self.current_nb_line = self.nb_lines - 1
                self.current_index_length -= 1

            # If it was the first point selected
            if self.current_index_length < 0:
                self.current_nb_line = 0
                self.current_index_length = 0

    def add_points(self, points):
        self.empty = False
        for (point2d_x, poind2d_y) in points:
            point3d_x = self.width_line * self.current_nb_line
            point3d_y = self.lengths[self.current_index_length]
            new_point = np.array([[point2d_x, poind2d_y, 1], [point3d_x, point3d_y, 0]])
            self.points[self.current_nb_line, self.current_index_length] = new_point

    def erase_point(self):
        self.back()
        if not self.empty:
            previous_point = self.points[self.current_nb_line, self.current_index_length, 1]
            expected_previous_x = self.width_line * self.current_nb_line
            expected_previous_y = self.lengths[self.current_index_length]
            expected_point = np.array([expected_previous_x, expected_previous_y, 0])

            if np.array_equal(previous_point, expected_point):
                self.points[self.current_nb_line, self.current_index_length] = np.zeros((2, 3))
            if self.current_index_length == 0 and self.current_nb_line == 0:
                self.empty = True

    def show_points(self, grid_points, erase=False):
        current_length = self.current_index_length + 1
        current_line = self.current_nb_line + 1
        qlabel = grid_points.itemAtPosition(current_line, current_length)

        point = self.points[self.current_nb_line, self.current_index_length]
        point_2d = "{},".format(point[0, 0: 2])
        point_3d = " {};".format(point[1])
        qlabel.widget().setText(point_2d + point_3d)

        if not erase:
            self.next()


def initialize_grid(grid_points, messages_points_manager):
    nb_lengths = messages_points_manager.nb_lengths
    lengths = messages_points_manager.lengths
    nb_lines = messages_points_manager.nb_lines

    for index_line in range(nb_lines):
        for index_length in range(nb_lengths):
            grid_points.addWidget(QLabel("[0. 0.], [0. 0. 0.];"), index_line + 1, index_length + 1)

    for index_length in range(1, nb_lengths + 1):
        grid_points.addWidget(QLabel("Length {} meters".format(lengths[index_length - 1])), 0, index_length)

    for index_line in range(1, nb_lines + 1):
        grid_points.addWidget(QLabel("Line {}".format(index_line - 1)), index_line, 0)


def one_selection(image, label, messages_points_manager, grid_points):
    if not messages_points_manager.finished:
        messages_points_manager.add_points(select_points(image, messages_points_manager.nb_points))
    # Update points
    messages_points_manager.show_points(grid_points)
    # Update instructions
    label.setText(messages_points_manager.get_message())


def withdraw_point(label, messages_points_manager, grid_points):
    messages_points_manager.erase_point()
    # Update points
    messages_points_manager.show_points(grid_points, erase=True)
    # Update instructions
    label.setText(messages_points_manager.get_message())


def skip_length(label, messages_points_manager):
    label.setText(messages_points_manager.next_length())


def define_points(image):
    message_points = MessageAndPoints()

    # Set application, window and layouts
    app = QApplication([])
    window = QWidget()
    layout_button = QHBoxLayout()
    layout = QVBoxLayout()
    grid = QGridLayout()

    # Set buttons and labels
    identify_button = QPushButton('Identify the point')
    withdraw_button = QPushButton('Go one step back and erase')
    skip_button = QPushButton('Skip length')
    instructions = QLabel(message_points.get_message())
    initialize_grid(grid, message_points)

    # Set connections
    identify_button.clicked.connect(functools.partial(one_selection, image, instructions, message_points, grid))
    withdraw_button.clicked.connect(functools.partial(withdraw_point, instructions, message_points, grid))
    skip_button.clicked.connect(functools.partial(skip_length, instructions, message_points))
    layout_button.addWidget(identify_button)
    layout_button.addWidget(withdraw_button)
    layout_button.addWidget(skip_button)
    layout.addLayout(layout_button)
    layout.addWidget(instructions)
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
