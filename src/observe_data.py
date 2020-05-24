from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid0"
PERCENTAGE = 0.875  # percentage of the training set
NB_CLASSES = 10
DATA_AUGMENTING = False

# What to see
SHOW_SETS = True
FOR_VISU = False
SHOW_REPARTITION_TRAIN = True
SHOW_REPARTITION_VAL = True
SHOW_BY_IMAGE = False  # Withdraw standardization in data loader before


# --- Parameters --- #
# Parameters to get the data
PATH_BEGIN = "../"
PATH_DATA = Path(PATH_BEGIN + "data/lanes/{}/".format(VIDEO_NAME))
PATH_LABEL = Path(PATH_BEGIN + "data/head_points/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE, for_visu=FOR_VISU)
TRAIN_SET = GENERATOR.train
VAL_SET = GENERATOR.valid
if SHOW_SETS:
    print("Training set")
    print(TRAIN_SET)
    print("Validation set")
    print(VAL_SET)

TRAIN_DATA = DataLoader(TRAIN_SET, PATH_DATA, batch_size=1, nb_classes=NB_CLASSES, data_augmenting=DATA_AUGMENTING)
VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=1, nb_classes=NB_CLASSES)

print("The training set is composed of {} images".format(len(TRAIN_SET)))
print("The validation set is composed of {} images".format(len(VAL_SET)))


if SHOW_REPARTITION_TRAIN and not FOR_VISU:
    LABEL_REPARTITION = np.zeros(NB_CLASSES)
    for (BATCH, LABELS) in TRAIN_DATA:
        LABEL_REPARTITION[int(LABELS[0])] += 1
    print("The repartition of the classes of the training set is :")
    print(LABEL_REPARTITION / len(TRAIN_SET))

if SHOW_REPARTITION_VAL and not FOR_VISU:
    LABEL_REPARTITION = np.zeros(NB_CLASSES)
    for (BATCH, LABELS) in VALID_DATA:
        LABEL_REPARTITION[int(LABELS[0])] += 1
    print("The repartition of the classes of the validation set is :")
    print(LABEL_REPARTITION / len(VAL_SET))


if SHOW_BY_IMAGE:
    for (BATCH, LABELS) in TRAIN_DATA:
        (b, g, r) = cv2.split(BATCH[0])  # get b,g,r
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img.astype(int))
        print(LABELS[0])
        plt.show()


# Percentage for the training set with FOR_VISU = False, to have the entire lane 8 : 0.8959
# Percentage for the training set with FOR_VISU = True, to have the entire lane 8 : 0.875
# The repartition of the classes of the training set, with data augmenting is : (FOR_VISU = False)
    # [0.03116147 0.08038244 0.12252125 0.14164306 0.12322946 0.13456091, 0.1296034  0.12535411 0.08498584 0.02655807]
# The repartition of the classes of the training set, without data augmenting is : (FOR_VISU = False)
    # [0.         0.00318697 0.06126062 0.14305949 0.12747875 0.13031161, 0.12818697 0.18661473 0.1621813  0.05771955]
# The repartition of the classes of the validation set is : (FOR_VISU = False)
    # [0.         0.         0.03544304 0.14936709 0.11898734 0.16962025, 0.12658228 0.21012658 0.12151899 0.06835443]
