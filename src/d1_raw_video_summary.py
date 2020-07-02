"""
This module updates the file list_videos.txt.
This text file contains all the name of the videos
that has to be in the file data/videos.
"""
from pathlib import Path
import os
from src.d0_utils.store_load_matrix import store_matrix


def update_video_file(path_videos, path_begin):
    """
    Update the file list_videos.txt that registers the name
    of the videos that should be in the file data/videos.
    """
    list_new_video = []
    for video in os.listdir(path_videos):
        if video[-4:] == ".mp4":
            list_new_video.append(path_begin / video)
    store_matrix(list_new_video, path_videos / "list_videos.txt")


if __name__ == "__main__":
    PATH_VIDEOS = Path("../../data/0_raw_videos")
    PATH_BEGIN = Path("../../data/0_raw_videos")
    update_video_file(PATH_VIDEOS, PATH_BEGIN)
