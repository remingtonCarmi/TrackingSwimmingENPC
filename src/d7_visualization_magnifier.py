"""
This script creates a video where the predicted LABELS are printed on the lanes.
"""
from pathlib import Path

# To manage the graphic
from src.d7_visualization.graphic_manager import GraphicManager

# To manage the video
from src.d7_visualization.video_manager import VideoManager

# To generate and load data
from src.d4_modelling_neural.loading_data.data_generator import generate_data
from src.d4_modelling_neural.loading_data.data_loader import DataLoader

# The model
from src.d4_modelling_neural.zoom_model import ZoomModel
from src.d4_modelling_neural.zoom_model_deep import ZoomModelDeep

# To slice the lanes
from src.d5_model_evaluation.slice_lane.slice_lanes import slice_lane

# To make a video from images
from src.d0_utils.store_load_data.make_video import make_video


def observe_models(data_param, models_param, model_evaluator, tries):
    """
    Observe the models behavior.

    Args:
        data_param (list): (video_name, lane_number, dimensions, scale)

        models_param (list): (model_type1, model_type2, number_trainings, nb_epochs, batch_sizes, window_sizes, recoveries)

        model_evaluator (function): function to evaluate the model.

        tries (string): says if the training is done on colab : tries = "" or on the computer : tries = "/tries".
    """
    # Unpack the variables
    (video_name, lane_number, dimensions, scale) = data_param
    (model_type1, model_type2, number_trainings, nb_epochs, batch_sizes, window_sizes, recoveries) = models_param

    # --- Set the paths --- #
    path_video = Path("data/1_raw_videos/{}.mp4".format(video_name))
    path_label = [Path("data/3_processed_positions{}/{}.csv".format(tries, video_name))]
    starting_data_path = Path("data/2_intermediate_top_down_lanes/lanes{}".format(tries))
    starting_calibration_path = Path("data/2_intermediate_top_down_lanes/calibration{}".format(tries))
    path_save_graphic = Path("reports/graphic_results{}/{}_{}_{}_{}_{}.jpg".format(tries, video_name, model_type1[1:], window_sizes[0], model_type2[1:], window_sizes[1]))

    path_weight_rough = Path("data/4_models_weights{}/magnifier{}".format(tries, model_type1))
    path_current_weight_rough = path_weight_rough / "window_{}_epoch_{}_batch_{}_{}.h5".format(window_sizes[0], nb_epochs[0], batch_sizes[0], number_trainings[0])

    path_weight_tight = Path("data/4_models_weights{}/magnifier{}".format(tries, model_type2))
    path_current_weight_tight = path_weight_tight / "window_{}_epoch_{}_batch_{}_{}.h5".format(window_sizes[1], nb_epochs[1], batch_sizes[1], number_trainings[1])

    # --- Generate and load the sets --- #
    data = generate_data(path_label, starting_data_path, starting_calibration_path, take_all=False, lane_number=lane_number)
    set = DataLoader(data, scale=scale, batch_size=1, dimensions=dimensions, standardization=True, augmentation=False, flip=True)
    set_visu = DataLoader(data, scale=scale, batch_size=1, dimensions=dimensions, standardization=False, augmentation=False, flip=False)

    print("The set is composed of {} images".format(len(data)))

    # --- Define the graphic manager --- #
    graphic_manager = GraphicManager(path_video, starting_calibration_path / "{}.txt".format(video_name), scale, len(set), dimensions[1])

    # --- Define the video manager --- #
    video_manager = VideoManager(len(set))

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
    (lanes, labels) = set[0]
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

    for (idx_batch, batch) in enumerate(set):
        (lanes, labels) = batch
        swimming_way = data[idx_batch, 3]

        # -- Get the predictions -- #
        (index_preds, index_regression_pred) = model_evaluator(model_rough, model_tight, lanes[0], labels[0], window_sizes, recoveries)
        print("Prediction tight", index_preds)
        print("Regression prediction", index_regression_pred)
        # Take the swimming way into account
        if swimming_way == -1:
            index_regression_pred = dimensions[1] - index_regression_pred
            index_preds = dimensions[1] - index_preds

        # -- For the graphic -- #
        frame_name = data[idx_batch, 0].parts[-1][: -4]
        graphic_manager.update(idx_batch, frame_name, index_preds, index_regression_pred, set_visu[idx_batch][1][0, 1])

        # -- For the video -- #
        video_manager.update(idx_batch, set_visu[idx_batch][0][0], index_preds, index_regression_pred)

    # --- Make the graphic --- #
    graphic_manager.make_graphic(path_save_graphic)

    # --- Make the video --- #
    print("Making the video...")
    destination_video = Path("data/5_model_output/videos{}".format(tries))
    name_predicted_video = "predicted_{}_{}_window_{}_{}_window_{}.mp4".format(video_name, model_type1[1:], window_sizes[0], model_type2[1:], window_sizes[1])
    make_video(name_predicted_video, video_manager.lanes_with_preds, destination=destination_video)
