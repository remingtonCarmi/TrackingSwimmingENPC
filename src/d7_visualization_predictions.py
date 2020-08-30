"""
This script creates a video where the predicted labels are printed on the original video clip.
"""
from pathlib import Path

# To generate and load data
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The model
from src.d4_modelling_neural.zoom_model import ZoomModel
from src.d4_modelling_neural.zoom_model_deep import ZoomModelDeep

# To slice the lanes
from src.d5_model_evaluation.slice_lane.slice_lanes import slice_lane

# To manage the predictions
from src.d7_visualization.prediction_memories import PredictionMemories

# To undo the perspective
from src.d0_utils.perspective_correction.undo_perspective import read_homography, get_original_image

# To extract the images
from src.d0_utils.extractions.extract_image import extract_image_video


def observe_model(data_param, models_param, model_evaluator, tries):
    """
    Observe the models behavior.

    Args:
        data_param (list): (video_name, lane_number, dimensions, scale, begin_time, end_time)

        models_param (list): (model_type1, model_type2, number_trainings, nb_epochs, batch_sizes, window_sizes, recoveries)

        model_evaluator (function): function to evaluate the model.

        tries (string): says if the training is done on colab : tries = "" or on the computer : tries = "/tries".

    Returns:
        prediction_memories (PredictionMemories): an object that contains the list of the predictions.
    """
    # Unpack the variables
    (video_name, lane_number, dimensions, scale, begin_time, end_time) = data_param
    (model_type1, model_type2, number_trainings, nb_epochs, batch_sizes, window_sizes, recoveries) = models_param

    # --- Set the paths --- #
    path_video = Path("data/1_raw_videos/{}.mp4".format(video_name))
    path_label = [Path("data/3_processed_positions{}/{}.csv".format(tries, video_name))]
    starting_data_path = Path("data/2_intermediate_top_down_lanes/lanes{}".format(tries))
    starting_calibration_path = Path("data/2_intermediate_top_down_lanes/calibration{}".format(tries))

    path_weight_rough = Path("data/4_models_weights{}/magnifier{}".format(tries, model_type1))
    path_current_weight_rough = path_weight_rough / "window_{}_epoch_{}_batch_{}_{}.h5".format(
        window_sizes[0], nb_epochs[0], batch_sizes[0], number_trainings[0]
    )

    path_weight_tight = Path("data/4_models_weights{}/magnifier{}".format(tries, model_type2))
    path_current_weight_tight = path_weight_tight / "window_{}_epoch_{}_batch_{}_{}.h5".format(
        window_sizes[1], nb_epochs[1], batch_sizes[1], number_trainings[1]
    )

    # --- Define the prediction memories --- #

    prediction_memories = PredictionMemories(
        begin_time,
        end_time,
        path_video,
        starting_calibration_path,
        dimensions,
        scale,
        extract_image_video,
        generate_data,
        DataLoader,
        read_homography,
        get_original_image,
    )

    # --- Generate and load the sets --- #
    data = generate_data(
        path_label, starting_data_path, starting_calibration_path, take_all=True, lane_number=lane_number
    )
    # Withdraw the frame that are out of the laps of time of interest
    data = prediction_memories.in_time(data)
    set_loader = DataLoader(
        data, batch_size=1, scale=scale, dimensions=dimensions, standardization=True, augmentation=False, flip=True
    )

    print("The set is composed of {} images".format(len(data)))

    # --- Define the MODELS --- #
    if model_type1 == "/deep_model":
        model_rough = ZoomModelDeep(False)
    else:
        model_rough = ZoomModel(False)
    if model_type2 == "/deep_model":
        model_tight = ZoomModelDeep(True)
    else:
        model_tight = ZoomModel(True)

    # --- Get the weights of the trainings --- #
    # Build the rough model to load the weights
    print(set_loader[0])
    (lanes, labels) = set_loader[0]
    print("lanes", lanes)
    print("labels", labels)
    (sub_lanes, sub_labels) = slice_lane(lanes[0], labels[0], window_sizes[0], recoveries[0])[:2]
    model_rough.build(sub_lanes.shape)
    # Load the weights
    model_rough.load_weights(str(path_current_weight_rough))

    # Build the tight model to load the weights
    (sub_lanes, sub_labels) = slice_lane(lanes[0], labels[0], window_sizes[1], recoveries[1])[:2]
    model_tight.build(sub_lanes.shape)
    # Load the weights
    model_tight.load_weights(str(path_current_weight_tight))

    # --- Evaluate the set --- #
    model_rough.trainable = False
    model_tight.trainable = False

    for (idx_batch, batch) in enumerate(set_loader):
        (lanes, labels) = batch
        swimming_way = data[idx_batch, 3]

        # -- Get the predictions -- #
        (index_preds, index_regression_pred) = model_evaluator(
            model_rough, model_tight, lanes[0], labels[0], window_sizes, recoveries
        )
        print("Prediction tight", index_preds)
        print("Regression prediction", index_regression_pred)

        # Take the swimming way into account
        if swimming_way == -1:
            index_regression_pred = dimensions[1] - index_regression_pred
            index_preds = dimensions[1] - index_preds

        # -- For the original video -- #
        frame_name = data[idx_batch, 0].parts[-1][:-4]
        prediction_memories.update(frame_name, index_preds[0], index_preds[-1], index_regression_pred)

    return prediction_memories
