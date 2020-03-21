"""
Allows to merge the lines-separated frames in a unique frame
"""

from crop_lines import *


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
    lines, points = load_lines("test0.jpg", "vid0", 0, 0)

    m = merge(lines)
    plt.imshow(m)
    plt.show()
