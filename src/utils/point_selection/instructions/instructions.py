"""
Gives the instruction to the user to tell the possibilities.
"""
from PyQt5.QtWidgets import QMessageBox, QApplication

# Information about the point selection
INSTRUCTIONS_SELECTION = "Select a point : left click. \n"
INSTRUCTIONS_SELECTION += "Zoom in : keep left click, move and release. \n"
INSTRUCTIONS_SELECTION += "Zoom out : click anywhere. \n"
INSTRUCTIONS_SELECTION += "Withdraw a point : press escape button. \n"
INSTRUCTIONS_SELECTION += "Quit : press space bar. \n"


# Information about the calibration points
INSTRUCTIONS_CALIBRATION = "YOU ARE ABOUT TO SELECT FOUR POINTS AND WRITE THEIR REAL COORDINATES ! \n \n"

INSTRUCTIONS_CALIBRATION += INSTRUCTIONS_SELECTION + "\n"

INSTRUCTIONS_CALIBRATION += "WARNINGS ! \n"
INSTRUCTIONS_CALIBRATION += "Only numbers are accepted to write the coordinates. \n"
INSTRUCTIONS_CALIBRATION += "The color of a selected point should have the same color as the written text. \n"


# Information about the head points
INSTRUCTIONS_HEAD = "YOU ARE ABOUT TO SELECT HEAD POINTS ! \n \n"

INSTRUCTIONS_HEAD += INSTRUCTIONS_SELECTION + "Skip a point (if you don't see the head) : press control \n \n"

INSTRUCTIONS_HEAD += "WARNINGS ! \n"
INSTRUCTIONS_HEAD += "Do not press the scpace bar twice it will not register the information "
INSTRUCTIONS_HEAD += "for the second image. \n"


def instructions_calibration():
    """
    Create a message to tell the possibilities to the user for the calibration point selection.
    """
    message = QMessageBox()
    message.setText(INSTRUCTIONS_CALIBRATION)
    message.exec_()


def instructions_head():
    """
    Create a message to tell the possibilities to the user.
    """
    app = QApplication([])
    message = QMessageBox()
    message.setText(INSTRUCTIONS_HEAD)
    message.show()
    app.exec_()
