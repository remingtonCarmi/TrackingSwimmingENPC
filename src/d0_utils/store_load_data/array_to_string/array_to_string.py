"""
This file convert an array to a string ready to be registered
in a txt or csv file.
"""
import numpy as np


def array_to_string(matrix):
    """
    Convert an array or a list to a string ready to be registered.

    Args:
        matrix (array or list of 1 or 2 dimensions): the matrix to store.

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


if __name__ == "__main__":
    # -- Doc tests -- #
    import doctest
    doctest.testmod()
