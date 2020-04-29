from pathlib import Path
import csv
import numpy as np
import pandas as pd
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, FindError


def last_line(csv_path):
    # Verify that the csv file exists
    if not csv_path.exists():
        raise FindError(csv_path)

    # Get the last line
    last = pd.read_csv(csv_path).tail(1)
    if not last.empty:
        frame = int(last['frame'])
        lane = int(last['lane'])
    else:
        frame = -1
        lane = -1
    return frame, lane


def fill_csv(csv_name, list_points, list_image_name, destination_path=Path("../../../output/test/")):
    # Check that the folder exists
    if not destination_path.exists():
        raise FindError(destination_path)

    csv_path = destination_path / csv_name
    nb_points = len(list_points)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        first_line = ["frame", "lane", "x_head", "y_head"]
        writer.writerow(first_line)
        for idx_point in range(nb_points):
            name_point = list_image_name[idx_point]
            point_information = list_points[idx_point]
            writer.writerow(np.concatenate((name_point, point_information)))


if __name__ == "__main__":
    CVS_NAME = "test0.csv"
    LIST_POINTS = np.array([[4, 4.9], [9, 2.9]])
    LIST_IMAGE_NAME = np.array([[0, 5], [34, 2]])
    try:
        fill_csv(CVS_NAME, LIST_POINTS, LIST_IMAGE_NAME)
        LAST_LINE = last_line(Path("../../../output/test/") / CVS_NAME)
        print(LAST_LINE)
    except FindError as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
