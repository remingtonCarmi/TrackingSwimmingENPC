from pathlib import Path
import csv
import numpy as np
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, FindError
from src.utils.store_load_matrix.array_to_string.array_to_string import array_to_string


def fill_csv(csv_name, list_points, list_image_name, destination_path=Path("../../../output/test/")):
    # Check that the folder exists
    if not destination_path.exists():
        raise FindError(destination_path)

    # Check that the file does not exist
    csv_path = destination_path / csv_name
    if csv_path.exists():
        raise AlreadyExistError(csv_path)

    nb_points = len(list_points)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        first_line = ['frame,lane,x_head,y_head']
        writer.writerow(first_line)
        for idx_point in range(nb_points):
            point_information = array_to_string(list_points[idx_point])
            name_point = list_image_name[idx_point]
            writer.writerow([name_point + "," + point_information])


if __name__ == "__main__":
    CVS_NAME = "test0.csv"
    LIST_POINTS = np.array([[4, 4.9], [9, 2.9]])
    LIST_IMAGE_NAME = np.array(["0,5", "34,2"])
    try:
        fill_csv(CVS_NAME, LIST_POINTS, LIST_IMAGE_NAME)
    except FindError as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
