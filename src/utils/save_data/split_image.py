import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.save_data.crop_lines import crop


def split_and_save(image, margin, destination, frame, nb_lines):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    list_y = [int(1/nb_lines * i * image.shape[0]) for i in range(nb_lines + 1)]
    list_images = crop(image, list_y, margin)
    for c in range(10):
        name = 'f' + str(frame) + '_c%d.jpg' % c
        cv2.imwrite(str(destination / name), list_images[c])


if __name__ == "__main__":
    NAME = "..\\..\\..\\test\\red_boxes\\fig1.jpg"
    PATH = Path("../../../output/test/")

    IMAGE = plt.imread(NAME)

    split_and_save(IMAGE, -10, PATH, 8, 10)
