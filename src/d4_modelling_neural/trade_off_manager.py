"""
This module contains the class TradeOffManager.
"""
import numpy as np
import matplotlib.pyplot as plt


class TradeOffManager:
    """
    The class that manages the trade off.
    """
    def __init__(self, first_trade_off, last_trade_off, nb_trade_off, window_size, nb_epochs, batch_size):
        """
        Construct the trade off manager.

        Args:
            first_trade_off (float): the first trade off value.

            last_trade_off (float): the last trade off value.

            nb_trade_off (integer): the number of trade offs.

            window_size (integer): the width of the sub-images.

            nb_epochs (integer): the number of epochs that has been done.

            batch_size (integer): the size of the batch taken.
        """
        # The trade offs
        self.trade_offs = np.geomspace(first_trade_off, last_trade_off, nb_trade_off)

        # To name of the figure that will be stored
        self.save_name = "window_{}_epoch_{}_batch_{}.jpg".format(window_size, nb_epochs, batch_size)

        # The lists that will be plotted
        self.error_rates_train = []
        self.maes_train = []
        self.error_rates_valid = []
        self.maes_valid = []

    def __len__(self):
        return len(self.trade_offs)

    def __getitem__(self, item):
        """
        Returns a trade off with geometrical step.
        """
        return self.trade_offs[item]

    def add(self, training_information):
        """
        Add the training information to the lists.

        Args:
            training_information (list of 4 floats): accuracy on train, mae on train, accuracy on validation, mae on validation.
        """
        self.error_rates_train.append(1 - training_information[0])
        self.maes_train.append(training_information[1])
        self.error_rates_valid.append(1 - training_information[2])
        self.maes_valid.append(training_information[3])

    def save(self, path_root):
        """
        Save the 4 lists.

        Args:
            path_root (WindowsPath): the path that leads to the folder that will contain the figure.
        """
        plt.plot(self.trade_offs, self.error_rates_train, label="Error rates", color="blue")
        plt.plot(self.trade_offs, self.maes_train, label="Mean absolute errors", color="blue")
        plt.plot(self.trade_offs, self.error_rates_valid, label="Error rates", color="orange")
        plt.plot(self.trade_offs, self.maes_valid, label="Mean absolute errors", color="orange")
        plt.xlabel("Values of the trade off")
        plt.legend()
        plt.savefig(path_root / self.save_name)
        plt.close()
