"""
This module is where you can find all the exception classes.

Classes:
    FindPathExtractError
    TimeError
    EmptyFolder
    NoMoreFrame
"""


class FindPathExtractError(Exception):
    """The exception class error to tell that the path has not been found."""
    def __init__(self, path_name):
        """
        Construct the path_name.
        """
        self.path_name = path_name

    def __repr__(self):
        return "The path {} was not found.".format(self.path_name)


class TimeError(Exception):
    """The exception class error to tell that the time
    or the number of images is not possible"""
    def __init__(self, name, time_begin, time_end):
        """
        Args:
            name (string): name of the video.

            time_begin (integer in second): the first image will be at the second 'time'.

            time_end (integer in second): the final time at which we can_stop to register the video.
        """
        self.name = name
        self.time_begin = time_begin
        self.time_end = time_end

    def __repr__(self):
        """"Indicates that the time asked is not possible."""
        if self.time_begin > self.time_end:
            begin_message = "The begining time {} seconds is higher than".format(self.time_begin)
            end_message = " the ending time {} seconds.".format(self.time_end)
        else:
            begin_message = "The video {} is too short.".format(self.name)
            end_message = " Please lower the beginning time {} seconds".format(self.time_begin)
        return begin_message + end_message


class EmptyFolder(Exception):
    """The exception class error to tell that the folder is empty"""
    def __init__(self, path_folder):
        """
        Construct the path_folder.
        """
        self.path_folder = path_folder

    def __repr__(self):
        return "The folder {} is empty.".format(self.path_folder)


class NoMoreFrame(Exception):
    """The exception class error to tell that there is no more frame to point"""
    def __init__(self, path_folder):
        """
        Construct the path_folder.
        """
        self.path_folder = path_folder

    def __repr__(self):
        return "All the frames in {} have been labeled.".format(self.path_folder)
