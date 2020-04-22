from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow
from PyQt5.QtWidgets import QMessageBox, QLayout
from PyQt5.QtCore import Qt, QPoint, QRect, QSize

INSTRUCTIONS = "YOU ARE ABOUT TO SELECT FOUR POINTS AND WRITE THEIR REAL COORDINATES ! \n \n"

INSTRUCTIONS += "Select a point : left click. \n"
INSTRUCTIONS += "Zoom in : keep left click, move and release. \n"
INSTRUCTIONS += "Zoom out : click anywhere. \n"
INSTRUCTIONS += "Withdraw a point : press escape button. \n"
INSTRUCTIONS += "Quit : press space bar. \n \n"

INSTRUCTIONS += "WARNINGS ! \n"
INSTRUCTIONS += "Only numbers are accepted to write the coordinates. \n"
INSTRUCTIONS += "The color of a selected point should have the same color as the written text. \n"


def instructions():
    message = QMessageBox()
    message.setText(INSTRUCTIONS)
    message.exec_()
