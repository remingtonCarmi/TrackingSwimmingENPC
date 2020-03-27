"""
This code calibrates an entire video by withdrawing the distortion and the perspectives.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.extract_image import extract_image_video
from src.distortion.distortion import find_distortion_charact, clear_image, SelectionError
from src.extract_image import TimeError
from src.perspective.correction_perspective import correct_perspective_img
from src.detection import select_points


vid0 = "..\\data\\videos\\vid0"


class VideoError(Exception):
    """The exception class error to tell that the video does not exist."""
    def __init__(self, name_video):
        """
        Args:
            name_video (string): the name of the video
        """
        self.name_video = name_video

    def __repr__(self):
        """"Indicates that the video cannot be found."""
        return "The video {} cannot be found.".format(self.name_video)


def make_video(name_video, images):
    """
    Makes a video with all the images in images.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the images.
    """
    height, width, layers = images[0].shape
    size = (width, height)
    out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    print(name_video)
    for image in images:
        out.write(image)
    out.release()


def calibrate_video(name_video, time_begin, time_end, destination_video=""):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        name_video (string): name of the video.

        time_begin (integer): the starting time in second.

        time_end (integer): the ending time in second.

        destination_video (string): the destination path of the cleaned video.
    """
    # Get the images
    list_images = extract_image_video(name_video, time_begin, time_end, False)
    nb_images = len(list_images)
    if nb_images == 0:
        raise VideoError(name_video)
    list_unwarp_images = [0] * nb_images
    list_clean_images = [0] * nb_images

    # Get the caracteristics
    # points = select_points(list_images[0])
    points, charact = find_distortion_charact(list_images[0])
    src = np.float32([(points[0][0], points[0][1]),
                      (points[1][0], points[1][1]),
                       (points[3][0], points[3][1]),
                       (points[2][0], points[2][1])
            ])
    dst2 = np.float32([(1500, 0),
                  (0, 0),
                  (1500, 750),
                  (0, 750)])
    print(src)
    list_unwarp_images[0] = correct_perspective_img(list_images[0], src, dst2, True, False)
    print("taille originale: ", list_images[0].shape)
    print("taille modifiée: ", list_unwarp_images[0].shape)
    
    list_clean_images[0] = clear_image(list_unwarp_images[0], charact)

    for index_image in range(1, nb_images):
        list_unwarp_images[index_image] = correct_perspective_img(list_images[index_image], src, dst2, True, False)
        list_clean_images[index_image] = clear_image(list_unwarp_images[index_image], charact)

    name_video_clean = destination_video + name_video + "_clean.mp4"
    make_video(name_video_clean, list_unwarp_images)


if __name__ == "__main__":
    try:
        calibrate_video(vid0, 12, 15)
    except TimeError as time_error:
        print(time_error.__repr__())
    except SelectionError as select_error:
        print(select_error.__repr__())
    except VideoError as video_error:
        print(video_error.__repr__())
