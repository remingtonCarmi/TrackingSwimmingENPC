"""
Gives the instruction to the user to tell the possibilities.
"""
from PyQt5.QtWidgets import QMessageBox

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
    """
    Create a message to tell the possibilities to the user.
    """
    message = QMessageBox()
    message.setText(INSTRUCTIONS_CALIBRATION)
    message.exec_()
