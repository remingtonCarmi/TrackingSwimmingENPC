import numpy as np


# --- Transformation for the images --- #
def standardize(image):
    """
    Normalize between -1 and 1.
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


# --- Transformation for the labels --- #
def transform_label(label, nb_classes, image_size):
    length_class = image_size / nb_classes

    return int(label[0] // length_class)


# --- Data augmenting --- #
def augmenting(images, labels, random_seed, data_manager, nb_classes):
    if random_seed == 0:
        return data_manager.apply_transform(images, {"channel_shift_intensity": 10}), labels
    if random_seed == 1:
        return data_manager.apply_transform(images, {"flip_horizontal": True}), nb_classes - labels - 1
    if random_seed == 2:
        return data_manager.apply_transform(images, {"flip_vertical": True}), labels
    if random_seed == 3:
        return images, labels
    else:
        image = data_manager.apply_transform(images, {"flip_vertical": True})
        return data_manager.apply_transform(image, {"flip_horizontal": True}), nb_classes - labels - 1


if __name__ == "__main__":
    IMAGE = np.array([[[19, 3, 0], [12, 3, 2]], [[10, 31, 2], [2, 23, 28]]])
    LABEL = np.array([450, 12])

    print(standardize(IMAGE))
    print(transform_label(LABEL, 10, 1000))
