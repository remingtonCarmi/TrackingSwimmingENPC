import functools
import numpy as np
from src.utils.point_selection import select_points
from pathlib import Path
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QMessageBox, QVBoxLayout
from PyQt5.QtWidgets import QLabel


class MessageAndPoints:
    def __init__(self):
        self.points = []
        self.width_line = 2.5
        self.nb_points = 1

        self.finished = False
        self.nb_lines = 11
        self.lengths = [0, 5, 15, 35, 45, 50]
        self.nb_lengths = len(self.lengths)
        self.message = np.array((self.nb_lines, self.nb_lengths))

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
        for (point2d_x, poind2d_y) in points:
            point3d_x = self.width_line * self.current_nb_line
            point3d_y = self.lengths[self.current_index_length]
            self.points.append([(point2d_x, poind2d_y), (point3d_x, point3d_y, 0)])
        self.next()

    def erase_point(self):
        self.back()
        if len(self.points) > 0:
            previous_point = self.points[-1][1]
            expected_previous_x = self.width_line * self.current_nb_line
            expected_previous_y = self.lengths[self.current_index_length]
            if previous_point == (expected_previous_x, expected_previous_y, 0):
                self.points.pop()


def one_selection(image, label, messages_points_manager):
    if not messages_points_manager.finished:
        messages_points_manager.add_points(select_points(image, messages_points_manager.nb_points))
    label.setText(messages_points_manager.get_message())


def withdraw_point(label, messages_points_manager):
    messages_points_manager.erase_point()
    label.setText(messages_points_manager.get_message())


def skip_length(label, messages_points_manager):
    label.setText(messages_points_manager.next_length())


def show_points(messages_points_manager):
    alert = QMessageBox()
    alert.setText(str(messages_points_manager.points))
    alert.exec_()


def define_points(image):
    message_points = MessageAndPoints()

    # Set application, window and layouts
    app = QApplication([])
    window = QWidget()
    layout_button = QHBoxLayout()
    layout = QVBoxLayout()

    # Set buttons
    identify_button = QPushButton('Identify the point')
    withdraw_button = QPushButton('Withdraw last selection')
    skip_button = QPushButton('Skip length')
    show_button = QPushButton('Show points')
    information = QLabel(message_points.get_message())

    # Set connections
    identify_button.clicked.connect(functools.partial(one_selection, image, information, message_points))
    withdraw_button.clicked.connect(functools.partial(withdraw_point, information, message_points))
    skip_button.clicked.connect(functools.partial(skip_length, information, message_points))
    show_button.clicked.connect(functools.partial(show_points, message_points))
    layout_button.addWidget(identify_button)
    layout_button.addWidget(withdraw_button)
    layout_button.addWidget(skip_button)
    layout_button.addWidget(show_button)
    layout.addLayout(layout_button)
    layout.addWidget(information)

    # Show the buttons
    window.setLayout(layout)
    window.show()
    app.exec_()


if __name__ == "__main__":
    # print("Click left to select the point, press w to withdraw the last point, press q to exit.")
    IMAGE_PATH = Path("../../output/test/vid0_frame2.jpg")
    IMAGE = cv2.imread(str(IMAGE_PATH))
    define_points(IMAGE)
