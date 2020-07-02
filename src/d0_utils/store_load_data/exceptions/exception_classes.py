"""
This module is where you can find all the exception classes link with the storage of data.

Classes:
    FindPathExtractError
    AlreadyExistError
    NothingToAddError
"""


class FindPathError(Exception):
    """The exception class error to tell that the path has not been found."""
    def __init__(self, path_name):
        """
        Construct the path_name.
        """
        self.path_name = path_name

    def __repr__(self):
        return "The path {} was not found.".format(self.path_name)


class AlreadyExistError(Exception):
    """The exception class error to tell that the file already exists."""
    def __init__(self, path):
        """
        Construct the txt_name.
        """
        self.path = path

    def __repr__(self):
        return "The file {} already exists.".format(self.path)


class NothingToAddError(Exception):
    """The exception class error to tell that nothing is to add to the csv file."""
    def __init__(self, path_name):
        """
        Construct the path_name.
        """
        self.path_name = path_name

    def __repr__(self):
        return "No point where given to add to {}.".format(self.path_name)
