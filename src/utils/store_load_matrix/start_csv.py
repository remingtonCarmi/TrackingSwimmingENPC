"""
This module has the purpose of storing a matrix in a scv file.
"""
from pathlib import Path
import csv
import numpy as np


class CSVExistError(Exception):
    """The exception class error to tell that the csv file already exists."""
    def __init__(self, csv_name):
        """
        Construct the video_name.
        """
        self.csv_name = csv_name

    def __repr__(self):
        return "The file {} already exists.".format(self.csv_name)


def array_to_string(matrix):
    if isinstance(matrix, list):
        str_matrix = str(matrix)
    else:
        str_matrix = str(matrix.tolist())
    str_matrix = str_matrix.replace("\n", "")
    str_matrix = str_matrix.replace("[", "")
    str_matrix = str_matrix.replace("]", "")
    return [str_matrix.replace(" ", "")]


def store_calibration_csv(csv_name, data, destination_path=Path("../../../output/test/")):
    """
    Store the matrix in a .txt file.

    Args:
        destination_path (string): the complete path where the matrix will be stored.
    """
    print(data)
    csv_path = destination_path / csv_name
    if csv_path.exists():
        raise CSVExistError(csv_path)
    nb_line = len(data)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Register the video name
        writer.writerow([data[0]])
        for idx in range(1, nb_line):
            writer.writerow(array_to_string(data[idx]))


if __name__ == "__main__":
    DATA = [Path("SN.mp4").parts[0], np.array([[8.5, 0.], [1.5, 0.], [5.5, 1.]]), [[5.2, 0.], [4.4, 0.], [8.5, 9.]]]
    try:
        store_calibration_csv("test5.csv", DATA)
    except CSVExistError as exist_error:
        print(exist_error.__repr__())
