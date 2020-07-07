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
