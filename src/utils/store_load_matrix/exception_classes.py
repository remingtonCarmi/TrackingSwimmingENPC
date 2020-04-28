class AlreadyExistError(Exception):
    """The exception class error to tell that the file already exists."""
    def __init__(self, path):
        """
        Construct the txt_name.
        """
        self.path = path

    def __repr__(self):
        return "The file {} already exists.".format(self.path)

class FindError(Exception):
    """The exception class error to tell that the path has not been found."""
    def __init__(self, path_name):
        """
        Construct the path_name.
        """
        self.path_name = path_name

    def __repr__(self):
        return "The path {} was not found.".format(self.path_name)