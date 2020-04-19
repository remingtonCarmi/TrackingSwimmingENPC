from pathlib import Path
import cv2


def make_video(name_video, images, fps=25, destination=Path("../output/test/")):
    """
    Makes a video with all the images in images.

    Args:
        name_video (string): the name of the video.

        images (list of array of 3 dimensions - height, width, layers): list of the images.
    """
    height, width, layers = images[0].shape
    path_video = destination / name_video
    size = (width, height)
    out = cv2.VideoWriter(str(path_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for image in images:
        out.write(image)
    out.release()
