"""
This module is where you can find all the exception classes.

"""


class FolderAlreadyExists(Exception):
    """The exception class error to tell that the folder we want to create does already exist """
    def __init__(self, path_folder):
        """
        Construct the path_folder.
        """
        self.path_folder = path_folder

    def __repr__(self):
        return "The folder {} already exists.".format(self.path_folder)
