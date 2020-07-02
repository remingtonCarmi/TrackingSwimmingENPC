"""
This class construct a QWidget to manage the points selection.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt


class MainWidget(QWidget):
    """
    Highest class for points selection.
    """
    def __init__(self, index_close=1):
        super().__init__()
        self.setFocusPolicy(True)
        # The index of the child that is the image selection widget
        self.index_close = index_close

    def keyReleaseEvent(self, event):
        """
        If space bar is pressed, it closes the application.
        """
        if event.key() == Qt.Key_Space:
            self.close()

    def closeEvent(self, event):
        """
        Close the image selection widget in order to register the points.
        """
        children = self.children()
        children[self.index_close].close()

    def erase_point(self):
        """
        Call the erase method of the image selection widget.
        """
        self.children()[1].erase_point()
