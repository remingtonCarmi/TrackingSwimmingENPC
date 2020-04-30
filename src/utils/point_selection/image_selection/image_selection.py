"""
This file allows the user to select points in an image.
"""
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QPoint, QRect


class ImageSelection(QLabel):
    """
    QLabel class that shows an image and allows the user to point at pixels.

    Interaction events :
        - if the user click a point is selected
        - if the user click and drag, it zooms
        - if the user press the escape button, it erases the last point
        - if the user press the space bar, the application is closed.
        - if the user press the control button and that skip=True, the point is skip.
    """
    def __init__(self, pix_map, size, points, colors, skip=False, can_stop=False):
        """
        Constructs all the parameters to manager the QLabel.

        Args:
            pix_map (QPixmap): the image to display.

            size (QSize): the size of the QLabel.

            points (array, shape = (len(colors), 2)): the points that have not yet been selected.

            colors (list of QColors): the colors of the points.

            skip (optional)(boolean): if True, allows the user to skip points.

            can_stop (optional)(boolean): if True, allows the user to stop the pointing.
        """
        super().__init__()

        # Tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(True)

        # Points management
        self.list_point = np.zeros(len(colors), dtype=QPoint)  # The list that is updated with time
        self.points = points  # The list that is updated at the end
        self.register = False  # If the points have been registered
        self.nb_points = 0
        self.cursor_pos = QPoint()
        self.p_start = QPoint()
        self.p_end = QPoint()

        # Background management
        self.colors = colors
        self.real_image_size = pix_map.size()
        self.pix_map = pix_map.scaled(size, Qt.IgnoreAspectRatio)
        self.setPixmap(self.pix_map)
        self.setFixedSize(size)
        top_left = QPoint()
        bottom_right = QPoint(self.size().width() - 1, self.size().height() - 1)
        self.rect_in_image = QRect(top_left, bottom_right)
        self.zoom_in = False

        # Skip point option
        self.skip = skip

        # Stop point option
        self.can_stop = can_stop
        self.stop = False

    def paintEvent(self, event):
        """
        Calls draw_points.
        """
        super().paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        self.draw_points(painter)
        painter.end()

    def mouseMoveEvent(self, event):
        """
        Update the QLabel.
        """
        self.cursor_pos = event.pos()
        self.update()

    def mousePressEvent(self, event):
        """
        Remind the selected point.
        """
        self.p_start = event.pos()

    def mouseReleaseEvent(self, event):
        """
        Call update_points.
        """
        self.p_end = event.pos()
        self.update_points()
        self.update()

    def keyReleaseEvent(self, event):
        """
        Erase a point if escape is pressed.
        Skip a point if self.skip = True and control is pressed.
        Quit if space bar is pressed.
        """
        if event.key() == Qt.Key_Escape:
            self.erase_point()

        if self.skip and event.key() == Qt.Key_Control:
            # Skip point only if a point can be registered
            if self.nb_points < len(self.colors):
                self.skip_points()

        if self.can_stop and event.key() == Qt.Key_S:
            self.stop = True
            self.parentWidget().close()

        if event.key() == Qt.Key_Space:
            self.parentWidget().close()

    def closeEvent(self, event):
        """
        Register the points from list_point to points.
        """
        # closeEvent is called twice since the super needs to be closed
        if not self.register:
            self.register = True
            for index_point in range(self.nb_points):
                x_select = self.list_point[index_point].x()
                y_select = self.list_point[index_point].y()
                if x_select == -1:
                    x_image = -1
                    y_image = -1
                else:
                    x_image = int((x_select / self.size().width()) * self.real_image_size.width())
                    y_image = int((y_select / self.size().height()) * self.real_image_size.height())
                self.points[index_point] = np.array([x_image, y_image])

    def update_points(self):
        """
        Updates the points.
        if the mouse did not move :
            zoom out.
            add the point if it is possible.
        if the mouse did move:
            zoom in.
        """
        # Add a point if the mouse did not move
        if self.p_start == self.p_end:
            # There should be has much as selected points as colors
            if self.nb_points < len(self.colors):
                # Withdraw the zoom and add the point
                if self.zoom_in:
                    # Withdraw the zoom
                    self.setPixmap(self.pix_map.scaled(self.size(), Qt.IgnoreAspectRatio))
                    self.zoom_in = False
                    # Add the point
                    self.list_point[self.nb_points] = self.point_in_image()
                    # Reset the rectangle
                    top_left = QPoint()
                    bottom_right = QPoint(self.size().width() - 1, self.size().height() - 1)
                    self.rect_in_image = QRect(top_left, bottom_right)
                else:
                    self.list_point[self.nb_points] = self.p_start
                self.nb_points += 1
        # Zoom if the mouse mouved
        else:
            self.zoom()
            self.zoom_in = True

    def draw_points(self, q_painter):
        """
        Draw a circle near the mouse and
        draw the points that have been selected.

        Args:
            q_painter (QPainter): the QPainter to paint.
        """
        # Draw a circle that follows the mouse
        q_painter.setPen(Qt.white)
        if not self.cursor_pos.isNull():
            q_painter.drawEllipse(self.cursor_pos.x() - 2, self.cursor_pos.y() - 2, 4, 4)

        # Draw the points
        for index_point in range(self.nb_points):
            q_painter.setPen(self.colors[index_point])
            x_point = self.list_point[index_point].x()
            y_point = self.list_point[index_point].y()
            if self.in_showed_image(x_point, y_point):
                (x_point, y_point) = self.adapt_point(x_point, y_point)
                q_painter.drawEllipse(x_point - 1, y_point - 1, 2, 2)
                q_painter.drawEllipse(x_point - 2, y_point - 2, 4, 4)
                q_painter.drawEllipse(x_point - 4, y_point - 4, 8, 8)

    def zoom(self):
        """
        Zoom in according to the movement of the mouse.
        """
        rect_in_screen = self.rectangle_in_screen()
        self.rect_in_image = self.rectangle_in_image(rect_in_screen)

        zoom_pix = self.pix_map.copy(self.rect_in_image)
        self.setPixmap(zoom_pix.scaled(self.size(), Qt.IgnoreAspectRatio))

    def rectangle_in_screen(self):
        """
        Compute the rectangle in the QWidget that has been selected.

        Returns:
            (QRect): the rectangle in the QLabel.
        """
        top_left_x = max(min(self.p_start.x(), self.p_end.x()), 0)
        top_left_y = max(min(self.p_start.y(), self.p_end.y()), 0)
        top_left = QPoint(top_left_x, top_left_y)

        bottom_right_x = min(max(self.p_start.x(), self.p_end.x()), self.size().width())
        bottom_right_y = min(max(self.p_start.y(), self.p_end.y()), self.size().height())
        bottom_right = QPoint(bottom_right_x, bottom_right_y)

        return QRect(top_left, bottom_right)

    def rectangle_in_image(self, rectangle_zoom):
        """
        Compute the rectangle in the image that has been selected.

        Args:
            rectangle_zoom (QRect): the rectangle in the QLabel that
                has been selected.

        Returns:
            (QRect): the rectangle in the image.
        """
        start_x = self.rect_in_image.topLeft().x()
        start_y = self.rect_in_image.topLeft().y()

        top_left_x = start_x + (rectangle_zoom.topLeft().x() / self.size().width()) * self.rect_in_image.width()
        top_left_y = start_y + (rectangle_zoom.topLeft().y() / self.size().height()) * self.rect_in_image.height()
        top_left = QPoint(int(top_left_x), int(top_left_y))

        bottom_right_x = start_x + (rectangle_zoom.bottomRight().x() / self.size().width()) * self.rect_in_image.width()
        bottom_right_y = start_y + (rectangle_zoom.bottomRight().y() / self.size().height()) * self.rect_in_image.height()
        bottom_right = QPoint(int(bottom_right_x), int(bottom_right_y))

        return QRect(top_left, bottom_right)

    def point_in_image(self):
        """
        Compute the corresponding point in the image.

        Returns:
            (QPoint): the point in the image.
        """
        start_x = self.rect_in_image.topLeft().x()
        start_y = self.rect_in_image.topLeft().y()

        point_x = start_x + (self.p_start.x() / self.size().width()) * self.rect_in_image.width()
        point_y = start_y + (self.p_start.y() / self.size().height()) * self.rect_in_image.height()

        return QPoint(int(point_x), int(point_y))

    def in_showed_image(self, x_coord, y_coord):
        """
        Say if x_coord and y_coord represent a point in the showed image.

        Args:
            x_coord (float): the horizontal coordinate.

            y_coord (float): the vertical coordinate.

        Returns:
            (boolean): True if the point (x_coord, y_coord) is in the showed image.
        """
        if self.rect_in_image.left() < x_coord < self.rect_in_image.right():
            if self.rect_in_image.top() < y_coord < self.rect_in_image.bottom():
                return True
        return False

    def adapt_point(self, x_coord, y_coord):
        """
        Compute the coordinate of the point (x_coord, y_coord) in the showed image.

        Args:
            x_coord (float): the horizontal coordinate.

            y_coord (float): the vertical coordinate.

        Returns:
            in_zoom_x (float): the horizontal coordinate in the zoom image.

            in_zoom_y (float): the vertical coordinate in the zoom image.
        """
        in_zoom_x = (x_coord - self.rect_in_image.topLeft().x()) * self.size().width() / self.rect_in_image.width()
        in_zoom_y = (y_coord - self.rect_in_image.topLeft().y()) * self.size().height() / self.rect_in_image.height()

        return in_zoom_x, in_zoom_y

    def erase_point(self):
        """
        Erase the last point and update the display.
        """
        # If a point has been selected
        if self.nb_points > 0:
            self.nb_points -= 1
        self.update()

    def skip_points(self):
        """
        Add a point (-1, -1) to the list.
        """
        self.list_point[self.nb_points] = QPoint(-1, -1)
        self.nb_points += 1
