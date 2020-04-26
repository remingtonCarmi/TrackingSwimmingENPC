"""
This class allows the user to put the information about some points.
"""
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout
from PyQt5.QtCore import QSize
from src.utils.point_selection.information_points.edit_point.edit_point import EditPoint
from src.utils.point_selection.information_points.text_point.text_point import TextPoint


class CalibrationPoints(QGridLayout):
    """
    QGridLayout class that displays QTextEdit objects.
    """
    def __init__(self, size, colors, points):
        """
        Construct the objects to manage the QTextEdit objects.

        Args:
            size (QSize): the size of the QGridLayout.

            colors (list of Qt.Color): the colors of the points.

            points (array, shape = (len(colors), 2)): the points on which the information
                will be registered.
        """
        super().__init__()
        self.colors = colors
        self.nb_points = len(colors)
        self.size = size
        self.size.setHeight(self.size.height() / (2 * self.nb_points))
        self.size_text = QSize(self.size.width() / 4, self.size.height())
        self.size_edit = QSize(self.size.width() / 4, self.size.height() / 4)
        self.points = points

        self.set_raw_labels()

    def set_raw_labels(self):
        """
        Fill the QGridLayout with QLabel and QTextEdit.
        """
        for index_color in range(self.nb_points):
            color = self.colors[index_color]
            point_layout = QHBoxLayout()

            color_point = TextPoint("Point {}".format(index_color + 1), self.size)

            edit_meter = EditPoint(self.size_edit, color, self.points, index_color, 0)
            point_layout.addWidget(edit_meter)

            text_meter = TextPoint("meters", self.size_text)
            point_layout.addWidget(text_meter)

            text_line = TextPoint("n° line", self.size_text)
            point_layout.addWidget(text_line)

            edit_line = EditPoint(self.size_edit, color, self.points, index_color, 1)
            point_layout.addWidget(edit_line)

            self.addWidget(color_point, 2 * index_color, 0)
            self.addLayout(point_layout, 2 * index_color + 1, 0)
