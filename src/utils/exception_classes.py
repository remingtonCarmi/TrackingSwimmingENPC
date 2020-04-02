"""
This module is where you can find all the exception classes.

Classes:
    TimeError
    VideoFindError
    VideoPresenceError
"""
from pathlib import Path


class TimeError(Exception):
    """The exception class error to tell that the time
    or the number of images is not possible"""
    def __init__(self, name, time_begin, time_end):
        """
        Args:
            name (string): name of the video.

            time_begin (integer in second): the first image will be at the second 'time'.

            time_end (integer in second): the final time at which we stop to register the video.
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
            begin_message = "The video {} is too short to capture the frames asked.".format(self.name)
            end_message = " Please lower the begining time {} seconds".format(self.time_begin)
        return begin_message + end_message


class VideoFindError(Exception):
    """The exception class error to tell that the video has not been found."""
    def __init__(self, video_name):
        """
        Construct the video_name.
        """
        self.video_name = video_name

    def __repr__(self):
        return "The video {} is was not found.".format(self.video_name)


class VideoPresenceError(Exception):
    """
    The error class to say that a video is missing when the video is required.
    """
    def __init__(self, video_name):
        """
        Construct the video name.
        """
        self.video_name = video_name

    def __repr__(self):
        """
        Print the error with the name of the video.
        """
        path_video = Path("../../data/videos/")
        return "The video {} is not in the file {}".format(self.video_name, path_video)
