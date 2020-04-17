from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QHBoxLayout, QTextEdit, QGridLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QWindow
from PyQt5.QtWidgets import QMessageBox, QLayout
from PyQt5.QtCore import Qt, QPoint, QRect, QSize


class TextPoint(QLabel):
    def __init__(self, text, size):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(size)
