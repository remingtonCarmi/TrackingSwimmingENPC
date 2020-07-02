"""
The class is used to show the index of the point.
"""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


class TextPoint(QLabel):
    """
    Shows the index of the point.
    """
    def __init__(self, text, size):
        """
        Construct the QLabel.

        Args:
            text (string): the index of the points.

            size (QSize): the size of the QLabel.
        """
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(size)
