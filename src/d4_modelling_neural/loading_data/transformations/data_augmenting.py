# --- Data augmenting --- #
def augmenting(image, label, random_seed, data_manager):
    if random_seed == 0:
        return data_manager.apply_transform(image, {"channel_shift_intensity": 10}), label
    if random_seed == 1:
        return data_manager.apply_transform(image, {"flip_horizontal": True}),  - label - 1
    if random_seed == 2:
        return data_manager.apply_transform(image, {"flip_vertical": True}), label
    if random_seed == 3:
        return image, label
    else:
        image = data_manager.apply_transform(image, {"flip_vertical": True})
        return data_manager.apply_transform(image, {"flip_horizontal": True}),  - label - 1
