"""
Allows to merge the lines-separated frames in a unique frame
"""

from crop_lines import *
import cv2


def bgr_to_rgb(image):
    im = image.copy()
    im[:, :, 0], im[:, :, 2] = image[:, :, 2], image[:, :, 0]
    return im


def merge(images):
    """

    @param images: list of numpy arrays
    @return:
    """

    merged_image = images[0]
    for image in images[1:]:
        merged_image = np.concatenate((merged_image, image), axis=0)
    return merged_image


if __name__ == "__main__":
    lines, points = load_lines("frame2.jpg", "vid0_clean", 0, 0)

    m = merge(lines)

    plt.imshow(bgr_to_rgb(m))
    plt.show()
