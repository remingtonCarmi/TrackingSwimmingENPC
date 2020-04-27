import numpy as np
import matplotlib.pyplot as plt


def y_size(rectangles, i, j):
    """ give the length along the y axis of the rectangle at indices i,j in rectangles"""
    return abs(rectangles[i, j, 1] - rectangles[i, j, 3])


def x_size(rectangles, i, j):
    """ give the length along the x axis of the rectangle at indices i,j in rectangles"""
    return abs(rectangles[i, j, 0] - rectangles[i, j, 2])


def x_front(rectangles, i, j):
    """ give the x position of the right size of the rectangle at indices i,j in rectangles"""
    return rectangles[i, j, 2]


def area(rectangles, i, j):
    """ give the area of the rectangle at indices i,j in rectangles"""
    return y_size(rectangles, i, j) * x_size(rectangles, i, j)


def plot_graphs(rect, rectangles_to_plot, parameter="x_front"):
    """
    To plot the evolution of dimensions of a list of rectangles
    Args:
        rect (list): list of rectangles. A rectangle is described by (x0, y0, x1, y1) with
            x0 (integer): the x-coordinate of the top left pixel of the rectangle
            y0 (integer): the y-coordinate of the top left pixel of the rectangle
            x1 (integer): the x-coordinate of the bottom right pixel of the rectangle
            y1 (integer): the y-coordinate of the bottom right pixel of the rectangle
        rectangles_to_plot (list of integers): rectangles to plot the evolution of
        parameter (string): the dimension to plot the evolution of. See the four functions below.
        smooth (bool): if True, graphs will be smoothed

    Returns:
        None

    """
    n = len(rect)
    x = np.arange(n)

    # transform to a numpy array
    rect_np = np.array(rect)

    dic = {"x_size": x_size,
           "y_size": y_size,
           "x_front": x_front,
           "area": area
           }

    for j in rectangles_to_plot:
        r = [dic[parameter](rect_np, i, j) for i in range(n)]

        plt.figure()
        plt.plot(x, r)

    plt.show()
