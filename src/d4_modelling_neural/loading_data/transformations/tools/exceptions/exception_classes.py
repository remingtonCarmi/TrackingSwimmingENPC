"""
This module is where you can find all the exception classes.

Classes:
    FindPathDataError
    PaddingError
    SwimmingWayError
"""


class FindPathDataError(Exception):
    """The exception class error to tell that the path has not been found."""
    def __init__(self, path_name):
        """
        Construct the path_name.
        """
        self.path_name = path_name

    def __repr__(self):
        return "The path {} was not found.".format(self.path_name)


class PaddingError(Exception):
    """The exception class error to tell that the padding to the image is impossible."""
    def __init__(self, image_dimensions, pad_dimensions):
        """
        Construct the image_dimensions and the pad_dimensions.
        """
        self.image_dimensions = image_dimensions
        self.pad_dimensions = pad_dimensions

    def __repr__(self):
        beginning = "The image with the dimensions : {}, {} ".format(self.image_dimensions[0], self.image_dimensions[1])
        middle = "cannot be pad. "
        end = "The padding dimensions are : {}, {}. ".format(self.pad_dimensions[0], self.pad_dimensions[1])
        return beginning + middle + end + "Please check the quality of the video."


class SwimmingWayError(Exception):
    """The exception class error to tell that the swimming way has not been registered."""
    def __init__(self, path_data_images):
        """
        Construct the name of the video.
        """
        self.name_video = path_data_images.parts[-1]

    def __repr__(self):
        return "The swimming way label of the video {} has not been registered.".format(self.name_video)
