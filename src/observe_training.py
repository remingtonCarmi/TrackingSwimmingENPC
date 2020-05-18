from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data_generation.data_loader import DataLoader
from src.data_generation.data_generator import DataGenerator
from src.networks.easy_model import EasyModel
from src.networks.hard_model import HardModel
from src.utils.visualization_deep import visualize


# --- TO MODIFY --- #
# Parameters for data
VIDEO_NAME = "vid1"
PERCENTAGE = 0.8  # percentage of the training set
FROM_COLAB = False
NB_CLASSES = 10

# Parameters for the training
NUMBER_TRAINING = 1
EASY_MODEL = True
DATA_AUGMENTING = False


# --- Parameters --- #
# Parameters to get the data
if FROM_COLAB:
    PATH_BEGIN = ""
else:
    PATH_BEGIN = "../"
PATH_DATA = Path(PATH_BEGIN + "output/test/{}/".format(VIDEO_NAME))
PATH_LABEL = Path(PATH_BEGIN + "output/test/{}.csv".format(VIDEO_NAME))


# --- Generate and load the data --- #
GENERATOR = DataGenerator(PATH_DATA, PATH_LABEL, percentage=PERCENTAGE)
VAL_SET = GENERATOR.valid
VALID_DATA = DataLoader(VAL_SET, PATH_DATA, batch_size=len(VAL_SET), nb_classes=NB_CLASSES)
(VALID_SAMPLES, VALID_LABELS) = VALID_DATA[0]
print("The validation set is composed of {} images".format(len(VALID_SAMPLES)))

# --- Define the MODEL --- #
if EASY_MODEL:
    MODEL = EasyModel(NB_CLASSES)
else:
    MODEL = HardModel(NB_CLASSES)
# Get the weights of the previous trainings
PATH_WEIGHT = Path(PATH_BEGIN + "trained_weights/")
if NUMBER_TRAINING > 0:
    # Build the model to load the weights
    MODEL.build(VALID_SAMPLES.shape)
    if EASY_MODEL:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "easy_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING - 1)
    else:
        PATH_FORMER_TRAINING = PATH_WEIGHT / "hard_model_nb_classes_{}_{}.h5".format(NB_CLASSES, NUMBER_TRAINING - 1)
    # Load the weights
    MODEL.load_weights(str(PATH_FORMER_TRAINING))


# --- Evaluation --- #
OUTPUTS = MODEL(VALID_SAMPLES)
PREDICTIONS = np.argmax(OUTPUTS, axis=1)


# --- Visualisation --- #
FRAMES = visualize(VAL_SET[:, 0], PREDICTIONS, PATH_DATA, NB_CLASSES)

for FRAME in FRAMES:
    plt.figure()
    plt.imshow(FRAME)

plt.show()
