"""
Allows to merge the lines-separated frames into a unique frame
"""

import matplotlib.pyplot as plt
import numpy as np


def merge(images):
    """
    To vertically merge 10 images into a single image. The first image will be on the top.
    Args:
        images (list of numpy arrays): list of images, all with the same length, not necessarily the same high

    Returns:
        merged_image(numpy array): the merged image
    """
    merged_image = images[0]
    for line in images[1:]:
        merged_image = np.concatenate((merged_image, line), axis=0)
    return merged_image


if __name__ == "__main__":
    from src.red_boxes.crop_lines import load_lines
    from src.bgr_to_rgb import bgr_to_rgb

    LINES, POINTS = load_lines("frame2.jpg", "vid0_clean", 0, 0)

    MERGED = merge(LINES)

    plt.imshow(bgr_to_rgb(MERGED))
    plt.show()
