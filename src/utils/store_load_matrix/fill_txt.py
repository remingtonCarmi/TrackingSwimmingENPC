"""
This module has the purpose of storing a matrix in a txt file.
"""
from pathlib import Path
import numpy as np


class TXTExistError(Exception):
    """The exception class error to tell that the txt file already exists."""
    def __init__(self, txt_name):
        """
        Construct the txt_name.
        """
        self.txt_name = txt_name

    def __repr__(self):
        return "The file {} already exists.".format(self.txt_name)


def array_to_string(matrix):
    """
    Convert an array or a list to a string ready to be registered.

    Args:
        matrix (array or list of 2 dimensions): the matrix to store.

    Returns:
        (string): the elements of matrix separated by a comma.

    >>> array_to_string(np.array([[3, 4],[8, 2]]))
    '3,4,8,2'
    >>> array_to_string([[3, 4],[8, 2]])
    '3,4,8,2'
    """
    if isinstance(matrix, list):
        str_matrix = str(matrix)
    else:
        str_matrix = str(matrix.tolist())
    str_matrix = str_matrix.replace("[", "")
    str_matrix = str_matrix.replace("]", "")

    return str_matrix.replace(" ", "")


def store_calibration_txt(txt_name, data, destination_path=Path("../../../output/test/")):
    """
    Store the matrix in a .txt file with the given data to allow to calibrate an image.

    Args:
        txt_name (string): the name of the .txt file that will be created.

        data (list, any length): the elements to put in the .txt file.

        destination_path (pathlib): the complete path where the matrix will be stored.
            Should lead to a folder.
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
    # -- Doc tests -- #
    import doctest
    doctest.testmod()

    DATA = [Path("SN.mp4").parts[0], np.array([[8.5, 0.], [1.5, 0.], [5.5, 1.]]), [[5.2, 0.], [4.4, 0.], [8.5, 9.]]]
    try:
        store_calibration_txt("test0.txt", DATA)
    except TXTExistError as exist_error:
        print(exist_error.__repr__())
