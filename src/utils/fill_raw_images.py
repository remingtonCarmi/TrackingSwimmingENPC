from pathlib import Path
import os
from src.utils.load_matrix import load_matrix
from src.utils.store_matrix import store_matrix


class VideoPresenceError(Exception):
    def __init__(self, video_name):
        self.video_name = video_name

    def __repr__(self):
        path_video = Path("../../data/videos/")
        return "The video {} is not in the file {}".format(self.video_name, path_video)


def update_video_file():
    list_new_video = []
    path_videos = Path("../../data/videos/")
    for video in os.listdir(path_videos):
        if video[-4:] == ".mp4":
            list_new_video.append(path_videos / video)
    store_matrix(list_new_video, path_videos / "list_videos.txt")


def fill_raw_images(list_videos):

    for file_name in list_videos:
        if not os.path.exists(file_name):
            raise VideoPresenceError(file_name)


if __name__ == "__main__":
    # update_video_file()
    PATH_VIDEOS = Path("../../data/videos/list_videos.txt")
    LIST_VIDEOS = load_matrix(PATH_VIDEOS, "path")
    try:
        fill_raw_images(LIST_VIDEOS)
    except VideoPresenceError as vid_pres:
        print(vid_pres.__repr__())
