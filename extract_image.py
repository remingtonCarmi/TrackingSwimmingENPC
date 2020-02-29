"""
This code allows the user to load an image that is in a video.
"""
import cv2


def extract_image_video(name_video, time, number_image=1):
    """
    Extracts number_image images from name_video and
    save them

    Args:
        name_video (string): name of the video

        time (interger): the first image will be at the second 'time'.
        We consider that there are 25 images per second.

        number_image (integer): number of images wanted
    """
    video = cv2.VideoCapture('{}.mp4'.format(name_video))
    nb_image_wait = time * 25
    count_image = 0

    # We find the first interesting image
    for i in range(nb_image_wait):
        (success, image) = video.read()

    # We register the interesting images
    while success and count_image < number_image:
        cv2.imwrite("frame%d.jpg" % count_image, image)
        (success, image) = video.read()
        print('Read a new frame: ', success)
        count_image += 1


if __name__ == "__main__":
    extract_image_video('vid0', 30)
