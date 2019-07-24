import json

import tensorflow as tf

from tensorflow.python.keras import Model

model = Model()

with open('train_parameters.json') as json_file:
    parameters = json.load(json_file)

with open(model_json_path) as file:
    json.dump(obj=model.to_json(), fp=file)
