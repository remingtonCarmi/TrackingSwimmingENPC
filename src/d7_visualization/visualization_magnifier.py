

def prediction(model, sub_lanes):
    predictions = model(sub_lanes)
    for pred in predictions