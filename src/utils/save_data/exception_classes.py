"""
This module is where you can find all the exception classes.

"""


class FileAlreadyExists(Exception):
    """The exception class error to tell that the file we want to create does already exist """
    def __init__(self, path_file):
        """
        Construct the path_folder.
        """
        self.path_file = path_file

    def __repr__(self):
        return "The file {} already exists.".format(self.path_file)
