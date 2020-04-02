"""
This module has the purpose of storing a matrix.
"""


def store_matrix(matrix, destination_path_name):
    """
    Store the matrix in a .txt file.
    Args:
        matrix (array, one dimension): the array to be stored.

        destination_path_name (string): the complete path where the matrix will be stored.
    """
    with open(destination_path_name, 'w') as file:
        for item in matrix:
            string_row = "{}".format(item)
            string_row += "\n"
            file.write(string_row)
