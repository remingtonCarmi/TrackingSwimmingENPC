"""
This script trains a MODEL with the magnifier concept.
"""
from pathlib import Path

# Exception
from src.d0_utils.store_load_data.exceptions.exception_classes import AlreadyExistError

# To generate the data
from src.d4_modelling_neural.loading_data.data_generator import generate_data

# To load the sets
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The models
from src.d4_modelling_neural.magnifier.zoom_model import ZoomModel

# To slice the LANES
from src.d4_modelling_neural.magnifier.slice_lane.slice_lanes import slice_lanes

# The loss
from src.d4_modelling_neural.magnifier.loss import get_loss, evaluate_loss

# The optimizer
from tensorflow.keras.optimizers import Adam

# The metric's manager
from src.d4_modelling_neural.magnifier.metrics import MetricsMagnifier


def train_magnifier(data_param, loading_param, training_param, tries):
    """
    Train the MODEL magnifier.

    Args:
        data_param (list): (video_names_train, video_names_valid, number_training, dimensions)

        loading_param (list): (scale, data_augmenting)

        training_param (list): (nb_epochs, batch_size, window_size, recovery, trade_off)

        tries (string): says if the training is done on colab : tries = "" or on the computer : tries = "/tries".
    """
    # Unpack the arguments
    (video_names_train, video_names_valid, number_training, dimensions) = data_param
    (scale, data_augmenting) = loading_param
    (nb_epochs, batch_size, window_size, recovery, trade_off) = training_param

    # -- Paths to the data -- #
    paths_label_train = []
    for video_name_train in video_names_train:
        paths_label_train.append(Path("data/2_processed_positions{}/{}.csv".format(tries, video_name_train)))

    paths_label_valid = []
    for video_name_valid in video_names_valid:
        paths_label_valid.append(Path("data/2_processed_positions{}/{}.csv".format(tries, video_name_valid)))

    starting_data_paths = Path("data/1_intermediate_top_down_lanes/LANES{}".format(tries))
    starting_calibration_paths = Path("data/1_intermediate_top_down_lanes/calibration{}".format(tries))

    # Verify that the weights path does not exists
    path_weight = Path("data/3_models_weights{}/magnifier".format(tries))
    path_new_weight = path_weight / "window_{}_{}.h5".format(window_size, number_training)

    if path_new_weight.exists():
        raise AlreadyExistError(path_new_weight)

    # --- Generate and load the sets--- #
    train_data = generate_data(paths_label_train, starting_data_paths, starting_calibration_paths)
    valid_data = generate_data(paths_label_valid, starting_data_paths, starting_calibration_paths)

    train_set = DataLoader(train_data, batch_size=batch_size, scale=scale, dimensions=dimensions, data_augmenting=data_augmenting)
    valid_set = DataLoader(valid_data, batch_size=batch_size, scale=scale, dimensions=dimensions, data_augmenting=data_augmenting)
    print("The training set is composed of {} images".format(len(train_data)))
    print("The validation set is composed of {} images".format(len(valid_data)))

    # --- Define the MODEL --- #
    model = ZoomModel()

    # Get the weights of the previous trainings
    if number_training > 1:
        # Get the input shape to build the MODEL
        # Build the MODEL to load the weights
        (lanes, labels) = train_set[0]
        # Get the sub images
        (sub_lanes, sub_labels) = slice_lanes(lanes, labels, window_size, recovery)

        # Build the MODEL
        model.build(sub_lanes.shape)

        path_former_training = path_weight / "window_{}_{}.h5".format(window_size, number_training - 1)

        # Load the weights
        model.load_weights(str(path_former_training))

    # Optimizer
    optimizer = Adam()

    # --- For statistics --- #
    metrics = MetricsMagnifier()

    # --- Training --- #
    for epoch in range(nb_epochs):
        # - Train on the train set - #
        print("Training, epoch n ° {}".format(epoch))
        model.trainable = True
        for (idx_batch, batch) in enumerate(train_set):
            (lanes, labels) = batch

            # Get the sub images
            (sub_lanes, sub_labels) = slice_lanes(lanes, labels, window_size, recovery)

            # Compute the loss, the gradients and the PREDICTIONS
            (grads, loss_value, predictions) = get_loss(model, sub_lanes, sub_labels, trade_off)

            # Optimize
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Register statistics
            metrics.update_loss(loss_value, len(sub_labels))
            metrics.update_acc(sub_labels[:, :2], predictions[:, :2])
            metrics.update_mae(sub_labels[:, 2], predictions[:, 2])
            metrics.update_nb_batches()

        print("The loss on training is {}".format(metrics.loss_train))

        # - Evaluate the validation set - #
        print("Validation, epoch n° {}".format(epoch))
        model.trainable = False
        for (idx_batch, batch) in enumerate(valid_set):
            (lanes, labels) = batch

            # Get the sub images
            (sub_lanes, sub_labels) = slice_lanes(lanes, labels, window_size, recovery)

            # Compute the loss value and the PREDICTIONS
            (loss_value, predictions) = evaluate_loss(model, sub_lanes, sub_labels, trade_off)

            # Register statistics
            metrics.update_loss(loss_value, len(sub_labels), train=False)
            metrics.update_acc(sub_labels[:, :2], predictions[:, :2], train=False)
            metrics.update_mae(sub_labels[:, 2], predictions[:, 2], train=False)
            metrics.update_nb_batches(train=False)

        print("The loss on validation is {}".format(metrics.loss_valid))

        # Update the metrics
        metrics.on_epoch_end()

        # Shuffle data
        train_set.on_epoch_end()

    # --- Save the weights --- #
    model.save_weights(str(path_new_weight))

    # To save the plots
    starting_path_save = Path("reports/figures_results/zoom_model{}".format(tries))
    metrics.save(starting_path_save, number_training)