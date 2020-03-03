"""
This code allows the user to load an image that is in a video.
"""
import cv2


class TimeError(Exception):
    """The exception class error to tell that the time
    or the number of images is not possible"""
    def __init__(self, time, name):
        """
        Args:
            time (integer): the time where we want to look to the video

            name (string): the name of the video we want to look at
        """
        self.time = time
        self.name = name

    def __repr__(self):
        """"Indicates the name of the video and the time asked"""
        begin_message = "The video {} is too short to capture the frames asked.".format(self.name)
        end_message = " Please lower the time {} seconds".format(self.time)
        return begin_message + end_message


def extract_image_video(name_video, time, number_image=1):
    """
    Extracts number_image images from name_video and
    save them.
    This raises an exception if the time or the number of images asked
    is not possible.

    Args:
        name_video (string): name of the video

        time (interger in second): the first image will be at the second 'time'.

        number_image (integer): number of images wanted
    """
    # Compute parameters
    video = cv2.VideoCapture('{}.mp4'.format(name_video))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nb_image_wait = int(time * fps)
    count_image = 0

    # Check if the time or the number of images asked is possible
    if nb_image_wait + number_image > frame_count:
        raise TimeError(time, name_video)

    # We find the first interesting image
    for i in range(nb_image_wait):
        (success, image) = video.read()

    # We register the interesting images
    while success and count_image < number_image:
        nb_count_image = nb_image_wait + count_image
        cv2.imwrite("frame%d.jpg" % nb_count_image, image)
        (success, image) = video.read()
        print('Read a new frame: ', success)
        count_image += 1


if __name__ == "__main__":
    try:
        extract_image_video('vid0', 30, 5)
    except TimeError as time_error:
        print(time_error.__repr__())
