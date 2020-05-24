from pathlib import Path
from src.vizualisation_red_boxes import animation_red_boxes
from src.utils.draw_rectangle.plot_evolution_graph import plot_graphs
from src.utils.draw_rectangle.exception_classes import VideoAlreadyExists
from src.utils.extractions.exceptions.exception_classes import TimeError
from src.utils.extractions.exceptions.exception_classes import FindErrorExtraction


PATH_VIDEO = Path("data/videos/vid0.mp4")

# lines for vid0. Here, we point it manually for better precision.
LINES = [115, 227, 336, 443, 550, 659, 763, 871, 981]

# number of lines of pixels we ignore for each lane
MARGIN = 15

try:
    RECTANGLES = animation_red_boxes(PATH_VIDEO, LINES, MARGIN, 11, 17, True,
                                     destination_video=Path("output/videos/"), path_txt=Path("data/calibration"))

    # lanes we want to plot the swim frequency
    LINES_TO_PLOT = [1]

    PARAMETER_TO_PLOT = "x_front"
    plot_graphs(RECTANGLES, LINES_TO_PLOT, PARAMETER_TO_PLOT)

except VideoAlreadyExists as already_exists:
    print(already_exists.__repr__())

except TimeError as time_error:
    print(time_error.__repr__())

except FindErrorExtraction as find_error:
    print(find_error.__repr__())
