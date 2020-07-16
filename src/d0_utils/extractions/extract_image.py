"""
This code allows the user to load a list of image that is in a video.
"""
from pathlib import Path
import cv2
from src.d0_utils.extractions.exceptions.exception_classes import TimeError, FindPathExtractError


def extract_image_video(path_video, time_begin=0, time_end=-1, register=False, destination=None):
    """
    Extracts number_image image from path_images and
    save them.
    This raises an exception if the duration is not possible regarding the video.
    If time_end is bigger than the duration of the video,
    the function register until the end

    Args:
        path_video (WindowsPath): path of the video.

        time_begin (integer in second): the first image taken will be at the second 'time'.

        time_end (integer in second): the final time at which we can_stop to register the video.
            if time_end == -1, the video is registered until the end.

        register (boolean): if True, the image will be registered.
            Default value = False

        destination (WindowsPath): the destination of the registered image.
            Default value = None.

    Returns:
        image (list of array of 3 dimensions: height, width, layers):
            list of the extracted image.
    """
    if destination is None:
        destination = Path("../../../data/2_intermediate_top_down_lanes/LANES/tries")

    # Verify if the video exists:
    if not path_video.exists():
        raise FindPathExtractError(path_video)

    # Verify if the destination exists:
    if register and not destination.exists():
        raise FindPathExtractError(destination)

    # Get the video and its characteristics
    video = cv2.VideoCapture(str(path_video))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nb_image_wait = int(time_begin * fps)
    images = []

    # We make sure that the video will be register until the end if time_end is
    # bigger than the duration of the video.
    if time_end == -1:
        time_end = frame_count / fps
    number_image = min(int(time_end * fps), frame_count) - int(time_begin * fps)

    # If time_begin == time_end, one picture is registered.
    if number_image == 0:
        number_image = 1

    # Check if the time or the number of image asked is possible
    if time_begin > time_end or nb_image_wait > frame_count:
        raise TimeError(path_video, time_begin, time_end)

    # We find the first interesting image
    video.set(cv2.CAP_PROP_POS_FRAMES, nb_image_wait)
    (success, image) = video.read()

    count_image = 0
    # We register the interesting image
    while success and count_image < number_image:
        images.append(image)
        nb_count_image = nb_image_wait + count_image
        if register:
            end_path = "{}_frame{}.jpg".format(path_video.parts[-1][: -4], nb_count_image)
            cv2.imwrite(str(destination / end_path), image)
        (success, image) = video.read()
        count_image += 1

    return images


if __name__ == "__main__":
    PATH_VIDEO = Path("../../../data/1_raw_videos/vid0.mp4")
    try:
        LIST_IMAGES = extract_image_video(PATH_VIDEO, time_begin=0, time_end=1, register=True)
    except TimeError as time_error:
        print(time_error.__repr__())
    except FindPathExtractError as find_error:
        print(find_error.__repr__())
