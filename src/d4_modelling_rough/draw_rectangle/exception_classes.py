"""
This module is where you can find all the exception classes.

"""


class VideoAlreadyExists(Exception):
    """The exception class error to tell that the video we want to create does already exist """
    def __init__(self, path):
        """
        Construct the path_folder.
        """
        self.path = path

    def __repr__(self):
        return "The video {} already exists.".format(self.path)
