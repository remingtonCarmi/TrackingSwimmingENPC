from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    # Path to the csv
    PATH_CSV = Path("../../../data/3_processed_positions/tries/vid0.csv")

    # Read the csv
    DATA = pd.read_csv(PATH_CSV)

    # Sort by index and save
    SORTED_DATA = DATA.sort_index()

    # -- Create the sorted csv file -- #
    # Create the keys
    dictionary = {'x_head': [], 'y_head': []}
    keys = pd.DataFrame(dictionary)

    # saving the dataframe
    keys.to_csv(PATH_CSV, index=False)

    # -- Add the sorted data to the csv file -- #
    with open(PATH_CSV, 'a', newline='') as csv_file:
        # Add data frame
        SORTED_DATA.to_csv(csv_file, header=False)
