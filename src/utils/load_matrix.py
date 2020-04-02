"""
This module has the purpose of storing a matrix.
"""
from pathlib import Path
import numpy as np


def load_matrix(name_path, dtype="float"):
    """
    Read the results at the end of the path.
    Args:
        name_path (string): the complete path where the matrix is stored.

        dtype (sting): indicates the dtype of the matrix.

    Returns:
        matrix (array, one dimension): the array that is stored at the end of the path.
    """
    table = []
    with open(name_path, 'r') as file:
        for line in file.readlines():
            line = line.split()
            if dtype == "float":
                table.append(np.float(line[0]))
            elif dtype == "path":
                table.append(Path(line[0]))
            else:
                table.append(Path(line[0]))
    return np.array(table)
