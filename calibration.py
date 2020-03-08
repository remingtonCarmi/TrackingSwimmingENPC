"""
This code calibrate an entire video by withdrawing the distortion and the perspectives.
"""
import cv2
from extract_image import extract_image_video
from distortion import find_distortion_charact, clear_image


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


def calibrate_video(name_video, time_begin, time_end):
    """
    Calibrates the video from the starting time to the end
    and register it.

    Args:
        name_video (string): name of the video.

        time_begin (integer): the starting time in second.

        time_end (integer): the ending time in second.
    """
    # Get the images
    list_images = extract_image_video(name_video, time_begin, time_end, False)
    nb_images = len(list_images)
    list_clean_images = [0] * nb_images
    cv2.imwrite("first_image.jpg", list_images[0])
    # Get the caracteristics
    charact = find_distortion_charact("first_image.jpg")
    list_clean_images[0] = clear_image(list_images[0], charact)
    for index_image in range(1, nb_images):
        list_clean_images[index_image] = clear_image(list_images[index_image], charact)

    name_video_clean = "test\\" + name_video + "_clean.mp4"
    make_video(name_video_clean, list_clean_images)
    print(nb_images)


if __name__ == "__main__":
    calibrate_video("vid0", 0, 2)
