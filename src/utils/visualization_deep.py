import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def darken_box(image, yc, xc, width, length):
    y1 = yc - length//2
    x1 = xc - width//2
    y2 = yc + length // 2
    x2 = xc + width // 2
    image[y1: y2 + 1, x1: x2 + 1, 1] -= 10
    image[y1: y2 + 1, x1: x2 + 1, 1] -= 40
    image[y1: y2 + 1, x1: x2 + 1, 2] -= 40
    return image


def visualize_prediction(image, predictions, nb_class=10, length=0.7):
    h, w = np.shape(image)[:-1]
    h_lane = h / 10
    length = int(h_lane * length)
    width = w // nb_class
    for i, prediction in enumerate(predictions):
        yc = int((i + 1 + 0.5) * h_lane)
        image = darken_box(image, yc, prediction, width, length)
    return image


def visualize_prediction_one_lane(image, prediction, nb_class=10, length=0.7):
    h, w = np.shape(image)[:-1]
    length = int(h * length)
    width = w // nb_class
    yc = int(0.5 * h)

    image = darken_box(image, yc, prediction, width, length)
    return image


if __name__ == '__main__':
    PATH_IMAGE = Path("../../output/test/visualization.jpg")
    IMAGE = np.copy(plt.imread(PATH_IMAGE))
    PREDICTIONS = [700 + np.random.randint(0, 50) for i in range(8)]
    IMAGE = visualize_prediction(IMAGE, PREDICTIONS)
    plt.imshow(IMAGE)

    PATH_IMAGE2 = Path("../../output/test/vid1/l1_f0008.jpg")
    IMAGE2 = np.copy(plt.imread(PATH_IMAGE2))
    PREDICTION = 700
    IMAGE2 = visualize_prediction_one_lane(IMAGE2, PREDICTION)
    plt.figure()
    plt.imshow(IMAGE2)

    plt.show()





