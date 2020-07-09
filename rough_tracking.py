"""
This script creates a video that is the same video as the input but with red boxes around swimmers.

To modify:
    PATH_VIDEO
    LINES
    MARGIN
    time_begin (11)
    time_end (19)
"""

from pathlib import Path
from src.d4_modelling_rough import animation_red_boxes
from src.d4_modelling_rough.draw_rectangle.plot_evolution_graph import plot_graphs
from src.d4_modelling_rough.draw_rectangle import VideoAlreadyExists
from src.d0_utils.extractions.exceptions import TimeError
from src.d0_utils.extractions.exceptions import FindErrorExtraction


PATH_VIDEO = Path("data/videos/vid0.mp4")

# lines for vid0. Here, we point it manually for better precision.
LINES = [115, 227, 336, 443, 550, 659, 763, 871, 981]

# number of lines of pixels we ignore for each lane_magnifier
MARGIN = 15

try:
    RECTANGLES = animation_red_boxes(PATH_VIDEO, LINES, MARGIN, 11, 19, True,
                                     destination_video=Path("output/videos/"), path_txt=Path("data/calibration"))

    # LANES we want to plot the swim frequency
    LINES_TO_PLOT = [1]

    PARAMETER_TO_PLOT = "x_front"
    plot_graphs(RECTANGLES, LINES_TO_PLOT, PARAMETER_TO_PLOT)

except VideoAlreadyExists as already_exists:
    print(already_exists.__repr__())

except TimeError as time_error:
    print(time_error.__repr__())

except FindErrorExtraction as find_error:
    print(find_error.__repr__())
