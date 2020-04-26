""" main for the red boxes animation"""


import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt
from pathlib import Path

from src.calibration import calibrate_video
from src.utils.extract_image import extract_image_video

from src.utils.crop_lines import crop_list

from src.utils.draw_rectangle.draw_rectangle import draw_rectangle
from src.utils.draw_rectangle.plot_evolution_graph import plot_graphs

from src.utils.make_video import make_video


from src.detection_with_colors.swimmer_detection import edges
from src.utils.draw_rectangle.extreme_active_pixels import extreme_white_pixels


def boxes_list(list_lines, list_y):
    rectangles = []
    for line, y in zip(list_lines, list_y):
        x0, y0, x1, y1 = extreme_white_pixels(edges(line, sigma=3, threshold=8)[:, :-150])
        y0 += y
        y1 += y

        # To check that we detected the swimmer but not sth else.
        # Boxes must be smaller than 1/8 of the swimming pool size.
        if abs(x1 - x0) < np.shape(list_lines[0])[1] / 8:
            rectangles.append([x0, y0, x1, y1])

        else:
            print("Detection of swimmer from line " + str(list_y.index(y) + 1) + " failed.")
            rectangles.append([0, 0, 0, 0])
    return rectangles


def boxes_list_images(list_images_crop, list_y):
    list_rectangles = []
    for list_lines in list_images_crop:
        list_rectangles.append(boxes_list(list_lines, list_y))
    return list_rectangles


def show_frames_failed(images, list_rectangles, lines=None, line=1):
    if lines is None:
        lines = [0, 1, 2, 3, 4, 5, 6, 7]
    detection_fails = plot_graphs(list_rectangles, lines)
    for i in detection_fails:
        plt.figure()
        plt.title("Image number " + str(i))
        box_i = list_rectangles[i][line]
        plt.imshow(
            images[i][box_i[1]: box_i[3], box_i[0]: box_i[2]])


def animation_red_boxes(path_video, is_calibrated, lines, margin, time_begin=0, time_end=-1, show_fails=False,
                        create_video=False, destination_video=Path("../output/videos/")):
    t = time()
    if is_calibrated:
        corrected_images = extract_image_video(path_video, time_begin, time_end)
    else:
        corrected_images = calibrate_video(path_video, time_begin, time_end, destination_video, True, False)

    n_images = len(corrected_images)

    for i in range(n_images):
        corrected_images[i] = cv2.cvtColor(corrected_images[i], cv2.COLOR_BGR2RGB)

    print("Swimmers detection...")
    list_images_crop = crop_list(corrected_images, lines, margin)
    list_rectangles = boxes_list_images(list_images_crop, lines)

    im = np.copy(corrected_images)

    for i in range(len(im)):
        for swimmer in list_rectangles[i]:
            im[i] = draw_rectangle(im[i],
                                   swimmer[0],
                                   swimmer[1] + margin,
                                   swimmer[2],
                                   swimmer[3] + margin,
                                   3)

    if show_fails:
        show_frames_failed(im, list_rectangles)

    if create_video:
        for i in range(n_images):
            im[i] = cv2.cvtColor(im[i], cv2.COLOR_RGB2BGR)

        name_video = path_video.parts[-1]

        # Get the fps
        video = cv2.VideoCapture(str(path_video))
        fps_video = int(video.get(cv2.CAP_PROP_FPS))

        # Make the video
        print("Make the animation video ...")
        corrected_video = "boxes_" + name_video
        make_video(corrected_video, im, fps_video, destination_video)

    print("Runtime : ", round(time() - t, 3), " seconds.")


if __name__ == "__main__":

    PATH_VIDEO = Path("../data/videos/vid0.mp4")
    PATH_VIDEO_CALIBRATED = Path("../output/videos/corrected_vid0.mp4")

    # lines for vid0
    LINES = [110, 227, 336, 443, 550, 659, 763, 871, 981]

    MARGIN = 15

    animation_red_boxes(PATH_VIDEO_CALIBRATED, True, LINES, MARGIN, 0, 11, False, True)