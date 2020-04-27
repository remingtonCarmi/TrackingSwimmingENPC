""" This code aims to associate the head coordinates with the corresponding images"""

from pathlib import Path
import cv2
from src.load_data import load_data


def associate_head_coordinates_with_image(image_path, head_coords):
    """

    :param
        image_path: (Path) path of the image
        data: (list of lists) contains on each line the frame number, the waterline number, the head coordinates
    :return: list containing the corresponding image and the head coordinates in the image
        image (array): the  corresponding image
        (x_h, y_h) (integers): the spatial coordinates of the head in the considered image

    """
    # checks if the if the image exists or not
    if not image_path.exists():
        raise ImageFindError(image_path)

    # Preprocessing: decompose the path name to recover the information (name of the video, frame and waterline numbers)
    name = str(image_path)
    name_list = name.split('\\')
    print(name_list)
    video_name = name_list[-2]
    image_name = name_list[-1]

    # checks if the video exists
    video_file = Path('/'.join(name_list[: len(name_list)-1]))
    if not video_file.exists():
        raise VideoFileError(video_name)

    list_image_name = image_name.split('_')
    frame_name = list_image_name[0]
    line_name = list_image_name[1]

    # check if the image is in the right format
    if not line_name.startswith('l') or not frame_name.startswith('f'):
        raise PathFormatError(image_path)

    # we find the frame number and the waterline number
    list_line_name = line_name.split('.')
    frame_number = frame_name[1:]
    line_number = list_line_name[0][1:]
    print("fnumber", frame_number)
    print("line number", line_number)

    # we find the coordinates of the head in the data
    [x_h, y_h] = [0, 0]
    for i in range(len(data)):
        if data[i][0] == int(frame_number) and data[i][1] == int(line_number):
            x_h = data[i][2]
            y_h = data[i][3]

    image = cv2.imread(str(image_path))
    return [image, (x_h, y_h)]


class PathFormatError(Exception):
    """This exception tells if the image_path is in the correct form ( path + 'f' + frame number + '_' + 'c' +
    line_number + '.jpg') """

    def __init__(self, image_path):
        """
        Args:
            image_path (path): name of the image.

        """
        self.path = str(image_path)
        path_image = str(image_path)
        path_image.split('/')
        self.image_name = path_image[-1]
        self.video_name = path_image[-2]

    @property
    def __repr__(self):
        """"Tests if the image path is in the right format and if the image exists"""
        message = ''
        if not self.image_name.startswith('f'):
            message = 'the image name {} has not the right format, ' \
                      'it does not contain the frame number'.format(self.image_name)
        else:
            image_char = self.image_name
            image_char.split('_')
            if not image_char[1].startswith('l'):
                message = 'the image name {} has not the right format, ' \
                          'it does not contain the waterline number'.format(self.image_name)
        return message


class ImageFindError(Exception):
    """tells that the image has not been found."""

    def __init__(self, image_name):
        """
        Constructs the image's name.
        """
        self.image_name = image_name

    def __repr__(self):
        return "The image {} is was not found.".format(self.image_name)


class VideoFileError(Exception):
    """tells that the corresponding video file was not found"""

    def __init__(self, video_name):
        """
        Constructs the image's name.
        """
        self.video_name = video_name

    def __repr__(self):
        video_path = Path("../../data/labels/")
        return "The video {} is not in the file {}".format(self.video_name, video_path)


if __name__ == "__main__":
    PATH = Path("../../data/labels/test_file.csv")
    IMAGE_PATH = Path("../../data/labels/f123_l4.jpg")
    data, coords_h = load_data(PATH)
    print(coords_h)
    print(associate_head_coordinates_with_image(IMAGE_PATH, data)[1])
