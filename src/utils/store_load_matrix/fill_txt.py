"""
This module has the purpose of storing a matrix in a txt file.
"""
from pathlib import Path
import numpy as np
from src.utils.store_load_matrix.exception_classes import AlreadyExistError, FindErrorStore
from src.utils.store_load_matrix.array_to_string.array_to_string import array_to_string


def store_calibration_txt(txt_name, data, destination_path=Path("../../../output/test/")):
    """
    Store the matrix in a .txt file with the given data to allow to calibrate an image.

    Args:
        txt_name (string): the name of the .txt file that will be created.

        data (list, any length): the elements to put in the .txt file.

        destination_path (pathlib): the complete path where the matrix will be stored.
            Should lead to a folder.
    """
    # Check that the folder exists
    if not destination_path.exists():
        raise FindErrorStore(destination_path)

    # Verify if the txt does not exist
    txt_path = destination_path / txt_name
    if txt_path.exists():
        raise AlreadyExistError(txt_path)

    nb_line = len(data)
    with open(txt_path, 'w') as file:
        # Register the video name
        file.write(data[0] + "\n")
        for idx in range(1, nb_line):
            file.write(array_to_string(data[idx]) + "\n")


if __name__ == "__main__":
    DATA = [Path("SN.mp4").parts[0], np.array([[8.5, 0.], [1.5, 0.], [5.5, 1.]]), [[5.2, 0.], [4.4, 0.], [8.5, 9.]]]
    try:
        store_calibration_txt("test1.txt", DATA)
    except FindErrorStore as find_error:
        print(find_error.__repr__())
    except AlreadyExistError as exist_error:
        print(exist_error.__repr__())
