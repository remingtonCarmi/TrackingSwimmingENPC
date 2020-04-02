import functools
from src.utils.point_selection import select_points
from pathlib import Path
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QMessageBox
from PyQt5.QtWidgets import QLabel


def one_selection(image, nb_points, points):
    points.append(select_points(image, nb_points))


def show_points(points):
    alert = QMessageBox()
    alert.setText(str(points))
    alert.exec_()


def define_points(image):
    point_3d = []
    message = ""
    for line_number in range(11):
        for length in [0, 5, 15, 35, 45, 50]:
            message += "Select the point on the line {} at the lenght {} meters.".format(line_number, length)

    # Set application, window and layout
    app = QApplication([])
    window = QWidget()
    layout = QHBoxLayout()

    # Set buttons
    next_button = QPushButton('Next')
    difficult_button = QPushButton('Difficult to identify')
    skip_button = QPushButton('Skip length')
    show_button = QPushButton('Show points')

    # Set connections
    next_button.clicked.connect(functools.partial(one_selection, image, 2, point_3d))
    show_button.clicked.connect(functools.partial(show_points, point_3d))
    layout.addWidget(next_button)
    layout.addWidget(difficult_button)
    layout.addWidget(skip_button)
    layout.addWidget(show_button)

    # Show the buttons
    window.setLayout(layout)
    window.show()
    app.exec_()


if __name__ == "__main__":
    # print("Click left to select the point, press w to withdraw the last point, press q to exit.")
    IMAGE_PATH = Path("../../output/test/vid0_frame2.jpg")
    IMAGE = cv2.imread(str(IMAGE_PATH))
    define_points(IMAGE)
