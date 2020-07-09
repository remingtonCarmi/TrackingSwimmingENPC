""" red boxes animation"""
import cv2
import os
import numpy as np
from time import time
from pathlib import Path

from src.d2_intermediate_calibration import calibrate_video
from src.d0_utils.calibration_from_txt import calibrate_from_txt

from src.d0_utils.split_and_save_data.crop.crop_lines import crop_list

from src.d4_modelling_rough.draw_rectangle.draw_rectangle import draw_rectangle
from src.d4_modelling_rough.draw_rectangle.plot_evolution_graph import plot_graphs
from src.d4_modelling_rough.draw_rectangle.exception_classes import VideoAlreadyExists

from src.d0_utils.extractions.exceptions.exception_classes import TimeError, FindPathExtractError

from src.d0_utils.store_load_data.make_video import make_video


from src.d4_modelling_rough.swimmer_detection import edges
from src.d4_modelling_rough.draw_rectangle.extreme_active_pixels import extreme_white_pixels


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
            # print("Detection of swimmer from line " + str(list_y.index(y) + 1) + " failed.")
            rectangles.append([0, 0, 0, 0])
    return rectangles


def boxes_list_images(list_images_crop, list_y):
    list_rectangles = []
    for list_lines in list_images_crop:
        list_rectangles.append(boxes_list(list_lines, list_y))
    return list_rectangles


def animation_red_boxes(path_video, lines, margin, time_begin=0, time_end=-1, create_video=False,
                        destination_video=Path("../output/tries/"), path_txt=Path("../data/calibration")):
    if create_video:
        name_video = path_video.parts[-1]
        corrected_video = "boxes_" + name_video
        if os.path.exists(str(destination_video / corrected_video)):
            raise VideoAlreadyExists(corrected_video)

    t = time()

    whole_path_txt = str(path_txt / "{}.txt".format(path_video.parts[-1][: -4]))
    if os.path.exists(whole_path_txt):
        corrected_images = calibrate_from_txt(path_video, whole_path_txt, time_begin, time_end)
    else:
        print("Calibration parameters file not found. Please calibrate the video manually.")
        corrected_images = calibrate_video(path_video, time_begin, time_end, destination_video,
                                           destination_txt=Path(path_txt), create_txt=True)

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

    if create_video:
        for i in range(n_images):
            im[i] = cv2.cvtColor(im[i], cv2.COLOR_RGB2BGR)

        # Get the fps
        video = cv2.VideoCapture(str(path_video))
        fps_video = int(video.get(cv2.CAP_PROP_FPS))

        # Make the video
        print("Make the animation video ...")
        make_video(corrected_video, im, fps_video, destination_video)
        print("The video has been created with success! Check", corrected_video)

    print("Process finished. Runtime : ", round(time() - t, 3), " seconds.")

    return list_rectangles


if __name__ == "__main__":

    PATH_VIDEO = Path("../data/videos/vid0.mp4")

    # lines for vid0
    LINES = [115, 227, 336, 443, 550, 659, 763, 871, 981]

    # number of lines of pixels we ignore for each lane_magnifier
    MARGIN = 15

    try:
        RECTANGLES = animation_red_boxes(PATH_VIDEO, LINES, MARGIN, 12, 13, True)

        # LANES we want to plot the swim frequency
        LINES_TO_PLOT = [0, 1, 2]

        PARAMETER_TO_PLOT = "x_front"
        plot_graphs(RECTANGLES, LINES_TO_PLOT, PARAMETER_TO_PLOT)

    except VideoAlreadyExists as already_exists:
        print(already_exists.__repr__())

    except TimeError as time_error:
        print(time_error.__repr__())

    except FindPathExtractError as find_error:
        print(find_error.__repr__())


