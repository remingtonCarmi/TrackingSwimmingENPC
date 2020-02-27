"""
This code allows the user to load an image that is in a video.
"""
import cv2


def extract_image_video(name_video, number_image=1):
    """
    Extracts number_image images from name_video and
    save them

    Args:
        name_video (string): name of the video

        number_image (integer): number of images wanted
    """
    video = cv2.VideoCapture('{}.mp4'.format(name_video))
    (success, image) = video.read()
    count_image = 0

    while success and count_image < number_image:
        cv2.imwrite("frame%d.jpg" % count_image, image)
        (success, image) = video.read()
        print('Read a new frame: ', success)
        count_image += 1


if __name__ == "__main__":
    extract_image_video('vid0')
