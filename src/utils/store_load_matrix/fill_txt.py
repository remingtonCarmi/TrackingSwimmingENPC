"""
This module has the purpose of storing a matrix in a scv file.
"""
from pathlib import Path
import numpy as np


class TXTExistError(Exception):
    """The exception class error to tell that the csv file already exists."""
    def __init__(self, txt_name):
        """
        Construct the video_name.
        """
        self.txt_name = txt_name

    def __repr__(self):
        return "The file {} already exists.".format(self.txt_name)


def array_to_string(matrix):
    if isinstance(matrix, list):
        str_matrix = str(matrix)
    else:
        str_matrix = str(matrix.tolist())
    str_matrix = str_matrix.replace("[", "")
    str_matrix = str_matrix.replace("]", "")

    return str_matrix.replace(" ", "")


def store_calibration_txt(txt_name, data, destination_path=Path("../../../output/test/")):
    """
    Store the matrix in a .txt file.

    Args:
        destination_path (string): the complete path where the matrix will be stored.
    """
    txt_path = destination_path / txt_name
    if txt_path.exists():
        raise TXTExistError(txt_path)
    nb_line = len(data)
    with open(txt_path, 'w') as file:
        # Register the video name
        file.write(data[0] + "\n")
        for idx in range(1, nb_line):
            file.write(array_to_string(data[idx]) + "\n")


if __name__ == "__main__":
    DATA = [Path("SN.mp4").parts[0], np.array([[8.5, 0.], [1.5, 0.], [5.5, 1.]]), [[5.2, 0.], [4.4, 0.], [8.5, 9.]]]
    try:
        store_calibration_txt("test0.txt", DATA)
    except TXTExistError as exist_error:
        print(exist_error.__repr__())
