"""
This module is initializing the folder data/images/raw_images
with one image of each video that should be in videos.
If one video is missing, an error is raised.
The file list_videos.txt contains all the videos
that has to be in the file data/videos.
"""
from pathlib import Path
import os
from src.utils.load_matrix import load_matrix
from src.utils.store_matrix import store_matrix
from src.utils.extract_image import extract_image_video
from src.utils.exception_classes import VideoPresenceError


def update_video_file():
    """
    Update the file list_videos.txt that registers the name
    of the videos that should be in the file data/videos.
    """
    list_new_video = []
    path_videos = Path("../../data/videos/")
    for video in os.listdir(path_videos):
        if video[-4:] == ".mp4":
            list_new_video.append(path_videos / video)
    store_matrix(list_new_video, path_videos / "list_videos.txt")


def fill_raw_images(list_videos):
    """
    Fill the raw_images folder will one image for each video
    that are registered in the file list_videos.txt

    Args:
        list_videos (path): the path to the folder were all the videos are.
    """
    destination = Path("../../data/images/raw_images/")

    for video_name in list_videos:
        if not os.path.exists(video_name):
            raise VideoPresenceError(video_name)

        extract_image_video(video_name, 5, 5, register=True, destination=destination)


if __name__ == "__main__":
    # !!! The following line has to be uncommented only when a new video is added !!! #
    # update_video_file()
    PATH_VIDEOS = Path("../../data/videos/list_videos.txt")
    try:
        LIST_VIDEOS = load_matrix(PATH_VIDEOS, "path")
        fill_raw_images(LIST_VIDEOS)
    except VideoPresenceError as vid_presence:
        print(vid_presence.__repr__())
