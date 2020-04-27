class AlreadyExistError(Exception):
    """The exception class error to tell that the file already exists."""
    def __init__(self, path):
        """
        Construct the txt_name.
        """
        self.path = path

    def __repr__(self):
        return "The file {} already exists.".format(self.path)
