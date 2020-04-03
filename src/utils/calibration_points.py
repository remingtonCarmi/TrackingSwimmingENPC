import functools
import numpy as np
from src.utils.point_selection import select_points
from pathlib import Path
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QMessageBox, QVBoxLayout
from PyQt5.QtWidgets import QLabel


class MessageToPrint:
    def __init__(self):
        self.finished = False
        self.nb_lines = 11
        self.lengths = [0, 5, 15, 35, 45, 50]
        self.nb_lengths = len(self.lengths)
        self.message = np.array((self.nb_lines, self.nb_lengths))

        self.current_nb_line = 0
        self.current_index_lenght = 0

    def get_message(self):
        begin_message = "Select the point on the line {} ".format(self.current_nb_line)
        end_message = "at the lenght {} meters.".format(self.lengths[self.current_index_lenght])
        return begin_message + end_message

    def next(self):
        self.current_nb_line += 1

        if self.current_nb_line == self.nb_lines:
            self.current_nb_line = 0
            self.current_index_lenght += 1

        if self.current_index_lenght == self.nb_lengths:
            self.finished = True

        return self.get_message()

    def next_length(self):
        self.current_index_lenght += 1

        if self.current_index_lenght == self.nb_lengths:
            self.finished = True

        return self.get_message()

    def back(self):
        self.current_nb_line -= 1

        if self.current_nb_line < 0:
            self.current_nb_line = self.nb_lines - 1
            self.current_index_lenght -= 1

        # If it was the first point to select
        if self.current_index_lenght < 0:
            self.current_index_lenght = 0

        return self.get_message()


def one_selection(image, nb_points, points, label, messages):
    points.append(select_points(image, nb_points))
    label.setText(messages.next())


def withdraw_point(points, label, messages):
    points.pop()
    label.setText(messages.back())


def skip_length(label, messages):
    label.setText(messages.next_length())


def show_points(points):
    alert = QMessageBox()
    alert.setText(str(points))
    alert.exec_()


def next_point(label, message):
    label.setText(message)


def define_points(image):
    point_3d = []
    message = MessageToPrint()

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
    information = QLabel(message.get_message())

    # Set connections
    identify_button.clicked.connect(functools.partial(one_selection, image, 2, point_3d, information, message))
    withdraw_button.clicked.connect(functools.partial(withdraw_point, point_3d, information, message))
    skip_button.clicked.connect(functools.partial(skip_length, information, message))
    show_button.clicked.connect(functools.partial(show_points, point_3d))
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
