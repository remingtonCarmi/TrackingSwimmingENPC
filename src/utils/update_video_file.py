"""
This module updates the file list_videos.txt.
This text file contains all the name of the videos
that has to be in the file data/videos.
"""
from pathlib import Path
import os
from src.utils.store_load_matrix.store_matrix import store_matrix


def update_video_file():
    """
    Update the file list_videos.txt that registers the name
    of the videos that should be in the file data/videos.
    """
    list_new_video = []
    path_videos = Path("../../data/videos/")
    begin_path = Path("data/videos/")
    for video in os.listdir(path_videos):
        if video[-4:] == ".mp4":
            list_new_video.append(begin_path / video)
    store_matrix(list_new_video, path_videos / "list_videos.txt")


if __name__ == "__main__":
    update_video_file()
