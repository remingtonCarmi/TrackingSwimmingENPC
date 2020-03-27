"""
To get the video with all the rectangle boxes at each time
"""

from src.bgr_to_rgb import bgr_to_rgb

from src.red_boxes.crop_lines import load_lines
from src.red_boxes.merge_lines import merge
from src.red_boxes.red_spectrum import *

from src.calibration import make_video

import cv2
import numpy as np


def boxes1(name, margin):
    """
    Useless for now
    Args:
        name:
        margin:

    Returns:

    """
    l_frames, list_y = load_lines(name, "vid0_clean", 0, 0)
    frames = l_frames[0]
    rectangles = []

    for frame, y in zip(frames[1:-1], list_y[1:-1]):
        frame = bgr_to_rgb(frame)[margin:-margin, :]
        c, s = get_rectangle(keep_edges(frame, 2, False), [0, y])
        rectangles.append([c, s])

    image = merge(frames)
    image = bgr_to_rgb(image)
    plt.figure()
    plt.imshow(image)
    for r in rectangles:
        draw_rectangle(r[0], r[1], True)
    plt.show()


def boxes_list(, margin):
    l_frames, list_y = load_lines(name, "vid0_clean", 0, 0)
    frames = l_frames[0]
    rectangles = []

    for frame, y in zip(frames[1:-1], list_y[1:-1]):
        frame = bgr_to_rgb(frame)[margin:-margin, :]
        c, s = get_rectangle(keep_edges(frame, 2, False), [0, y])
        rectangles.append([c, s])

    image = merge(frames)
    image = bgr_to_rgb(image)
    plt.figure()
    plt.imshow(image)
    for r in rectangles:
        draw_rectangle(r[0], r[1], True)
    plt.show()


def boxes(name, folder, margin, time_begin, time_end):
    """
    From a given video (with corrected perspective), plot the red boxes containing the swimmers, and save this new video

    Args:
        name (string): generic name for the images that will be saved
        folder (string) : name of the folder where we save all the images
        margin (integer): the number of pixels we exclude from the water line before looking for the swimmer
        time_begin (float): starting time of the video we consider
        time_end (float): ending time of the video we consider

    Returns:
        rectangles(numpy array): information about red boxes plotted, for each water line, at each time.
            shape: 8*n*2*2, with n = number of frames in the video
            rectangles[i][j][0] : x,y (integers) : the coordinates of top left corner of the box
            rectangles[i][j][0] : a,b (integers) : the x_size and the y_size of the box

    """
    lines_per_frame, list_y = load_lines(name, folder, "vid0_clean", time_begin, time_end)
    rectangles = []
    images = []
    count = 0
    size_y = np.shape(lines_per_frame[0][0])[1]
    for frame in lines_per_frame:
        rectangles_per_frame = []
        for line, y in zip(frame[1:-1], list_y[1:-1]):
            line = bgr_to_rgb(line)[margin:-margin, :]
            c, s = get_rectangle(keep_edges(line, 2, False), [0, y])
            rectangles_per_frame.append([c, s])
        rectangles.append(rectangles_per_frame)

        image = merge(frame)
        image = bgr_to_rgb(image)
        plt.figure()

        fig = plt.imshow(image)

        for r in rectangles_per_frame:
            if abs(r[1][0]) < size_y / 4:
                draw_rectangle(r[0], r[1], True)

        # to remove the axes from the saved figure
        plt.axis('off')
        plt.margins(0, 0)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        # to save the figure (with red boxes)
        real_name = FOLDER + 'fig%d.jpg' % count
        plt.savefig(real_name, bbox_inches='tight', pad_inches=-0.1)
        plt.close()

        # to give a different name for the different figures
        count += 1

        # we read the images to next build a video with all images
        image_plot = cv2.imread(real_name)
        images.append(image_plot)

    make_video("test\\video_boxes.mp4", images)
    return rectangles


def plot_length_rectangles(rect):
    """
    NOT FINISHED
    To plot graphs of characteristics of the boxes
    Args:
        rect: list of information about rectangles, returns by the function "boxes", see below

    Returns:
        None

    """
    n = len(rect)
    x = np.arange(n)
    print(n)

    # transform to a numpy array
    rect_np = np.array(rect)

    for j in range(8):
        r = [rect_np[i, j, 1, 0] for i in range(n)]
        plt.figure()
        plt.plot(x, r)
    plt.show()


if __name__ == "__main__":
    FOLDER = "..\\..\\test\\red_boxes\\"
    RECT = boxes("frame2.jpg", FOLDER, 8, 0, 0.2)
    # plot_length_rectangles(RECT)



