from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow, QColor
from PyQt5.QtWidgets import QMessageBox, QLayout, QDesktopWidget
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import cv2
from src.utils.point_selection.information_points.calibration_points import CalibrationPoints
from src.utils.point_selection.image_selection.image_selection import ImageSelection
from src.utils.point_selection.instructions.instructions import instructions_perspective


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.children()[1].setFocus()
        if event.key() == Qt.Key_Space:
            self.close()

    def closeEvent(self, event):
        children = self.children()
        children[1].close()

    def erase_point(self):
        self.children()[1].erase_point()