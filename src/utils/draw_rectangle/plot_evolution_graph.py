import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline


def y_size(rectangles, i, j):
    return abs(rectangles[i, j, 1] - rectangles[i, j, 3])


def center(rectangles, i, j):
    return (rectangles[i, j, 0] + rectangles[i, j, 2]) / 2


def x_size(rectangles, i, j):
    return abs(rectangles[i, j, 0] - rectangles[i, j, 2])


def area(rectangles, i, j):
    return y_size(rectangles, i, j) * x_size(rectangles, i, j)


def plot_graphs(rect, lines_to_plot, element=x_size, smooth=True):
    """
    To plot graphs of characteristics of the boxes
    Args:
        smooth:
        lines_to_plot:
        element:
        rect: list of information about rectangles, returns by the function "boxes", see below

    Returns:
        None

    """
    n = len(rect)
    x = np.arange(n)

    # transform to a numpy array
    rect_np = np.array(rect)
    frames_to_check = []
    for j in lines_to_plot:
        r = [element(rect_np, i, j) for i in range(n)]

        for i in range(n):
            if r[i] > 5000:
                frames_to_check.append(i)

        x_new = np.linspace(min(x), max(x), 1000)

        plt.figure()
        if smooth:
            r_smooth = spline(x, r, x_new)
            plt.plot(x_new, r_smooth)
        else:
            plt.plot(x_new, x_new)

    return frames_to_check
