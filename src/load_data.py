""" Aims to load the data from the csv files containing the frame, the waterline number and head coordinates"""

import pandas as pd
import numpy as np
from pathlib import Path

from numpy.core.multiarray import ndarray


def load_data(name_file):
    """
    loading the data ans store them in lists
    
    :param 
        name_file: (string) the name of the file
    :return: 
        X: (list of lists) contains all the data
    """
    data_set = pd.read_csv(str(name_file))
    data = np.concatenate([data_set[['f', 'c', 'head_x', 'head_y']].to_numpy()], axis=1)
    coord: ndarray = np.concatenate([data_set[['head_x', 'head_y']].to_numpy()], axis=1)
    return data, coord


if __name__ == "__main__":
    PATH = Path("../../data/head_points/")
    data, head_coords = load_data(PATH)
    print(data)
    print(head_coords)