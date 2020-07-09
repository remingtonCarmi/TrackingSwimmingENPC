import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from src.d7_visualization.merge_lanes.merge_lanes import merge


def darken_box(image, yc, xc, width, length):
    y1 = yc - length//2
    x1 = xc
    y2 = yc + length // 2
    x2 = xc + width
    image[y1: y2 + 1, x1: x2 + 1, 1] -= 10
    image[y1: y2 + 1, x1: x2 + 1, 1] -= 40
    image[y1: y2 + 1, x1: x2 + 1, 2] -= 40
    return image


def visualize_one_frame(image, predictions, nb_class=10, length=0.7):
    h, w = np.shape(image)[:-1]
    h_lane = h / 10
    length = int(h_lane * length)
    width = w // nb_class
    for i, prediction in enumerate(predictions):
        yc = int((i + 1 + 0.5) * h_lane)
        xc = prediction * width
        image = darken_box(image, yc, xc, width, length)
    return image


def visualize_one_lane(image, prediction, nb_class=10, length=0.7):
    h, w = np.shape(image)[:-1]
    length = int(h * length)
    width = w // nb_class
    yc = int(0.5 * h)
    xc = prediction * width

    image = darken_box(image, yc, xc, width, length)
    return image


def sort_data(x, y):
    """ Sort names and predictions to be taken one entire frame (8 lanes in the right order) by one entire frame"""
    data = np.block([[x], [y]])
    data = data.T
    data = sorted(data, key=lambda i: (i[0][3:] + i[0][:3]))
    data = np.array(data).T
    x_sorted = data[0, :]
    y_sorted = data[1, :].astype(float)
    y_sorted = y_sorted.astype(int)
    return x_sorted, y_sorted


def visualize(names, predictions, path, nb_classes, length=0.7):
    """Create the list of the frames, annotated with the prediction, sorted correctly"""
    names, predictions = sort_data(names, predictions)
    nb_frames = int(len(names) / 8)
    frames = []

    for i in range(nb_frames):
        lanes = []
        for lane in range(8):
            index = i * 8 + lane

            prediction = predictions[index]
            image = np.copy(plt.imread(Path(path / names[index])))

            image = visualize_one_lane(image, prediction, nb_classes, length)
            lanes.append(image)

        frame = merge(lanes)
        frames.append(frame)
    return frames


def animation_one_lane(names, predictions, path, nb_classes, length=0.7):
    names, predictions = sort_data(names, predictions)
    nb_frames = int(len(names))
    frames = []

    for i in range(nb_frames):
        prediction = predictions[i]

        image = np.copy(plt.imread(Path(path / names[i])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if prediction != 0:
            image = visualize_one_lane(image, prediction, nb_classes, length)
        frames.append(image)

    return frames


if __name__ == '__main__':
    NB_CLASSES = 10

    # ------------one frame----------------
    # PATH_IMAGE = Path("../../output/tries/visualization.jpg")
    # IMAGE = np.copy(plt.imread(PATH_IMAGE))
    # PREDICTIONS = [np.random.randint(3, NB_CLASSES - 3) for i in range(8)]
    # IMAGE = visualize_one_frame(IMAGE, PREDICTIONS, nb_class=NB_CLASSES)
    # plt.imshow(IMAGE)

    # ------------one lane----------------
    # PATH_IMAGE2 = Path("../../output/tries/vid0/l1_f0008.jpg")
    # IMAGE2 = np.copy(plt.imread(PATH_IMAGE2))
    # PREDICTION = 4
    # IMAGE2 = visualize_one_lane(IMAGE2, PREDICTION, nb_class=NB_CLASSES)
    # plt.figure()
    # plt.imshow(IMAGE2)

    # ------------several frames----------------

    # run create_data_set.py with vid0 and time range = (0, 1)
    X = np.array(
        ['l1_f0001.jpg', 'l1_f0002.jpg',
         'l2_f0001.jpg', 'l2_f0002.jpg',
         'l3_f0001.jpg', 'l3_f0002.jpg',
         'l4_f0001.jpg', 'l4_f0002.jpg',
         'l5_f0001.jpg', 'l5_f0002.jpg',
         'l6_f0001.jpg', 'l6_f0002.jpg',
         'l7_f0001.jpg', 'l7_f0002.jpg',
         'l8_f0001.jpg', 'l8_f0002.jpg'
         ])
    X = np.array(
        ['l1_f0001.jpg', 'l1_f0002.jpg',
         'l1_f0003.jpg', 'l1_f0015.jpg',
         'l1_f0004.jpg', 'l1_f0014.jpg',
         'l1_f0005.jpg', 'l1_f0013.jpg',
         'l1_f0006.jpg', 'l1_f0012.jpg',
         'l1_f0001.jpg', 'l1_f0011.jpg',
         'l1_f0007.jpg', 'l1_f0010.jpg',
         'l1_f0008.jpg', 'l1_f0009.jpg'
         ])
    Y = np.array([np.random.randint(0, NB_CLASSES) for i in range(16)])

    PATH = Path("../../data/lanes/vid0/")

    FRAMES = visualize(X, Y, PATH, NB_CLASSES)

    for FRAME in FRAMES:
        plt.figure()
        plt.imshow(FRAME)

    plt.show()
