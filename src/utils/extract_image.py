"""
This code allows the user to load an image that is in a video.
"""
from pathlib import Path
import cv2
import os


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


def extract_image_video(name_video, time_begin, time_end, register="False", destination=""):
    """
    Extracts number_image images from name_video and
    save them.
    This raises an exception if the duration is not possible regarding the video.
    If time_end is bigger than the duration of the video,
    the function register until the end

    Args:
        name_video (string): name of the video.

        time_begin (integer in second): the first image will be at the second 'time'.

        time_end (integer in second): the final time at which we stop to register the video.

        register (boolean): if True, the images will be registered.

        destination (string): the destination of the registered images.

    Returns:
        images (list of array of 3 dimensions - height, width, layers):
        list of the registered images.
    """
    # Compute parameters
    video = cv2.VideoCapture(str(name_video))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nb_image_wait = int(time_begin * fps)
    images = []
    # We make sure that the video will be register until the end if time_end is
    # bigger than the duration of the video.
    number_image = min(int(time_end * fps), frame_count) - int(time_begin * fps)

    # If time_begin == time_end, one picture is registered.
    if number_image == 0:
        number_image = 1

    # Check if the time or the number of images asked is possible
    if time_begin > time_end or nb_image_wait > frame_count:
        raise TimeError(name_video, time_begin, time_end)

    # If time_begin == 0, the first picture is registered
    if time_begin == 0:
        nb_image_wait = 1
    # We find the first interesting image
    for i in range(nb_image_wait):
        (success, image) = video.read()

    if not success:
        raise VideoFindError(name_video)
    count_image = 1
    # We register the interesting images
    while success and count_image <= number_image:
        images.append(image)
        nb_count_image = nb_image_wait + count_image
        if register:
            end_path = "frame{}.jpg".format(nb_count_image)
            cv2.imwrite(str(destination / end_path), image)
        (success, image) = video.read()
        print('Read a new frame: ', success)
        count_image += 1

    return images


if __name__ == "__main__":
    try:
        LIST_IMAGES = extract_image_video(Path("../../data/videos/vid0.mp4"), 0, 0, True, Path("../../output/test/"))
    except TimeError as time_error:
        print(time_error.__repr__())
    except VideoFindError as find_error:
        print(find_error.__repr__())
