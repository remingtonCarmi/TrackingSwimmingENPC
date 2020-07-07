
from src.d4_modelling_neural.magnifier.image_magnifier.image_magnifier import ImageMagnifier


def slice_lanes(lanes, labels, window_size, recovery):
    nb_lanes = len(lanes)
    sub_lanes = []
    sub_labels = []
    for idx_lanes in range(nb_lanes):
        magnifier = ImageMagnifier(lanes[idx_lanes], labels[idx_lanes], window_size, recovery)

        for (sub_lane, sub_label) in magnifier:
            sub_lanes.append(sub_lane)
            sub_labels.append(sub_label)

    return sub_lanes, sub_labels
