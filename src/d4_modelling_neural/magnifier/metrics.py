"""
This module manage the metric during the training.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MetricsMagnifier:
    """
    The class of the metric manager.
    """
    def __init__(self):
        """
        Declaration of the variables.
        """
        # For the loss
        self.losses_train = []
        self.loss_train = 0
        self.losses_valid = []
        self.loss_valid = 0

        # For the accuracy
        self.accuracies_train = []
        self.acc_train = 0
        self.accuracies_valid = []
        self.acc_valid = 0

        # For the mean absolute error
        self.maes_train = []
        self.mae_train = 0
        self.maes_valid = []
        self.mae_valid = 0

        # Batch counter
        self.nb_batches_train = 0
        self.nb_batches_valid = 0

    def update_loss(self, loss_value, nb_samples, train=True):
        """
        Update the loss.

        Args:
            loss_value (float): the value of the loss.

            nb_samples (integer): the number of samples that correspond to the loss value.

            train (boolean): says if the update is done during training.
                Default value = True
        """
        if train:
            self.loss_train += loss_value / nb_samples
        else:
            self.loss_valid += loss_value / nb_samples

    def update_acc(self, labels, predictions, train=True):
        """
        Update the accuracy.

        Args:
            labels (array of integers): list of [is_in_image].

            predictions (array of (integer, integer): list of [pred_is_in_image, pred_is_not_in_image].

            train (boolean): says if the update is done during training.
                Default value = True
        """
        # Get the guesses from logits
        # if argmax(PREDICTIONS) = 0 the MODEL says the is a head in the ith sample
        # if argmax(PREDICTIONS) = 1 the MODEL says there is not a head in the ith sample
        pred_prob = 1 - np.argmax(predictions, axis=1)

        # Compute the accuracy
        accuracy = np.mean(labels[:, 0] == pred_prob)

        # Update the global accuracy
        if train:
            self.acc_train += accuracy
        else:
            self.acc_valid += accuracy

    def update_mae(self, labels, predictions, train=True):
        """
        Update the mean square error.

        Args:
            labels (array of integers: list of [column].
                column is the index of the column of pixel where the head is located.

            predictions(array of integers: list of [pred_column].
                column is the index of the column of pixel where the head is located.

            train (boolean): says if the update is done during training.
                Default value = True
        """
        # Withdraw the LABELS where there was no head
        only_labels = np.where(labels == -1, 0, labels)
        only_preds = np.where(labels == -1, 0, predictions)
        nb_heads = np.sum(np.where(labels == -1, 0, 1))

        # Compute the mean absolute error
        mae = np.sum(np.abs(only_labels - only_preds)) / nb_heads

        # Update the global accuracy
        if train:
            self.mae_train += mae
        else:
            self.mae_valid += mae

    def update_nb_batches(self, train=True):
        """
        Update the number of batch.

        Args:
            train (boolean): says if the update is done during training.
                Default value = True

        """
        if train:
            self.nb_batches_train += 1
        else:
            self.nb_batches_valid += 1

    def on_epoch_end(self):
        """
        Update the lists of metric.
        """
        # Update lists
        self.losses_train.append(self.loss_train / self.nb_batches_train)
        self.losses_valid.append(self.loss_valid / self.nb_batches_valid)
        self.accuracies_train.append(self.acc_train / self.nb_batches_train)
        self.accuracies_valid.append(self.acc_valid / self.nb_batches_valid)
        self.maes_train.append(self.mae_train / self.nb_batches_train)
        self.maes_valid.append(self.mae_valid / self.nb_batches_valid)

        # Update variables
        self.loss_train = 0
        self.loss_valid = 0
        self.acc_train = 0
        self.acc_valid = 0
        self.mae_train = 0
        self.mae_valid = 0

        self.nb_batches_train = 0
        self.nb_batches_valid = 0

    def save(self, starting_path, number_training):
        """
        Save the metrics.

        Args:
            starting_path (WindowsPath): the path where the metrics will be registered.

            number_training (integer): the index of the current training.
        """
        # Save the losses
        path_loss = starting_path / "loss_{}.jpg".format(number_training)
        plt.plot(self.losses_train, label="Loss on training set")
        plt.plot(self.losses_valid, label="Loss on validation set")
        plt.xlabel("Number of epoch")
        plt.legend()
        plt.savefig(path_loss)
        plt.close()

        # Save the accuracies
        path_accuracy = starting_path / "accuracy_{}.jpg".format(number_training)
        plt.plot(self.accuracies_train, label="Accuracy on training set")
        plt.plot(self.accuracies_valid, label="Accuracy on validation set")
        plt.xlabel("Number of epoch")
        plt.legend()
        plt.savefig(path_accuracy)
        plt.close()

        # Save the mean absolute error
        path_mae = starting_path / "mae_{}.jpg".format(number_training)
        plt.plot(self.maes_train, label="Mean absolute error on training set")
        plt.plot(self.maes_valid, label="Mean absolute error on validation set")
        plt.xlabel("Number of epoch")
        plt.legend()
        plt.savefig(path_mae)
        plt.close()


if __name__ == "__main__":
    METRICS = MetricsMagnifier()
    STARTING_PATH_SAVE = Path("../../../reports/figures_results/tries")

    LABELS = np.array([[0, 1, -1], [0, 1, -1], [1, 0, 10]], dtype=np.float)

    for epoch in range(2):
        # Training
        PRED_GOOD = np.array([[-1, 201, 933], [10, 29, -88], [110, -7, 15]], dtype=np.float)

        METRICS.update_loss(1., 2)
        METRICS.update_acc(LABELS[:, :2], PRED_GOOD[:, :2])
        METRICS.update_mae(LABELS[:, 2], PRED_GOOD[:, 2])
        METRICS.update_nb_batches()

        # Validation
        PRED_BAD = np.array([[1, -201, 933], [100, 2, -193], [1, -77, 190]], dtype=np.float)

        METRICS.update_loss(3, 2, train=False)
        METRICS.update_acc(LABELS[:, :2], PRED_BAD[:, :2], train=False)
        METRICS.update_mae(LABELS[:, 2], PRED_BAD[:, 2], train=False)
        METRICS.update_nb_batches(train=False)

        METRICS.on_epoch_end()

    METRICS.save(STARTING_PATH_SAVE, 1)
