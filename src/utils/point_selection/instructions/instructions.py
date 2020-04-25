from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow
from PyQt5.QtWidgets import QMessageBox, QLayout
from PyQt5.QtCore import Qt, QPoint, QRect, QSize

INSTRUCTIONS_CALIBRATION = "YOU ARE ABOUT TO SELECT FOUR POINTS AND WRITE THEIR REAL COORDINATES ! \n \n"

INSTRUCTIONS_CALIBRATION += "Select a point : left click. \n"
INSTRUCTIONS_CALIBRATION += "Zoom in : keep left click, move and release. \n"
INSTRUCTIONS_CALIBRATION += "Zoom out : click anywhere. \n"
INSTRUCTIONS_CALIBRATION += "Withdraw a point : press escape button. \n"
INSTRUCTIONS_CALIBRATION += "Quit : press space bar. \n \n"

INSTRUCTIONS_CALIBRATION += "WARNINGS ! \n"
INSTRUCTIONS_CALIBRATION += "Only numbers are accepted to write the coordinates. \n"
INSTRUCTIONS_CALIBRATION += "The color of a selected point should have the same color as the written text. \n"


def instructions_perspective():
    message = QMessageBox()
    message.setText(INSTRUCTIONS_CALIBRATION)
    message.exec_()
