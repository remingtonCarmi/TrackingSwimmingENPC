from pathlib import Path
import csv
import numpy as np
from src.utils.store_load_matrix.exception_classes import AlreadyExistError
from src.utils.store_load_matrix.array_to_string.array_to_string import array_to_string


def fill_csv(csv_name, list_points, destination_path=Path("../../../output/test/")):
    csv_path = destination_path / csv_name
    if csv_path.exists():
        raise AlreadyExistError(csv_path)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        first_line = ['frame_lane,horizontal_coord,vertical_coord']
        writer.writerow(first_line)
        for point in list_points:
            writer.writerow([array_to_string(point)])


if __name__ == "__main__":
    CVS_NAME = "test0.csv"
    LIST_POINTS = np.array([[[4, 4.9], [9, 2.9]]])
    try:
        fill_csv(CVS_NAME, LIST_POINTS)
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
