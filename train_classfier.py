#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

from jalef.preprocessing import BertPreprocessor
from jalef.layers import Bert

# set random seeds
np.random.seed(1234)
tf.set_random_seed(1234)

# Initialize session
sess = tf.Session()


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


def read_parameters(path: str = "parameters.json") -> dict:
    """
    Read the training parameters from the given file.
    :param path: path to the json file
    :return: the parameters as a dictionary
    """
    with open(path) as json_file:
        parameters = json.load(json_file)

    return parameters


def create_simple_bert_classifier(pretrained_model_path, output_size, n_layers_to_finetune, max_seq_len, num_classes):
    in_id = tf.keras.layers.Input(shape=(max_seq_len,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_len,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    # Instantiate the custom Bert Layer
    x = Bert(pretrained_model_path=pretrained_model_path,
             output_size=output_size,
             n_layers_to_finetune=n_layers_to_finetune,
             pooling=Bert.Pooling.ENCODER_OUT)(bert_inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation="relu"))(x)
    x = tf.keras.layers.Flatten()(x)

    # Build the rest of the classifier
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    pred = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model(inputs=bert_inputs, outputs=pred)


def get_training_data(parameters):
    bp = BertPreprocessor(max_sequence_length=parameters["dataset"]["max_seq_len"],
                          pretrained_model_path=parameters["bert"]["pretrained_model_path"])

    train = pd.read_csv(os.path.join(parameters["dataset"]["path"], "train.csv"))
    X_train, y_train = bp.fit_transform_classification(train["Text"], train["Label"])

    valid = pd.read_csv(os.path.join(parameters["dataset"]["path"], "validation.csv"))
    X_valid, y_valid = bp.fit_transform_classification(valid["Text"], valid["Label"])

    test = pd.read_csv(os.path.join(parameters["dataset"]["path"], "test.csv"))
    X_test, y_test = bp.fit_transform_classification(test["Text"], test["Label"])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def run_training(parameters):
    name = "dataset=" + parameters["dataset"]["name"] + "_intents=" + str(parameters["dataset"]["n_intents"])

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_training_data(parameters)

    model = create_simple_bert_classifier(pretrained_model_path=parameters["bert"]["pretrained_model_path"],
                                          output_size=parameters["bert"]["output_size"],
                                          n_layers_to_finetune=parameters["bert"]["n_layers_to_finetune"],
                                          max_seq_len=parameters["dataset"]["max_seq_len"],
                                          num_classes=len(y_train[0]))

    # Print layers
    model.summary()

    # Compile model
    model.compile(optimizer=parameters["hyperparameters"]["optimizer"], 
                  loss=parameters["hyperparameters"]["loss"], 
                  metrics=parameters["hyperparameters"]["metrics"])

    # Save model configuration
    with open(os.path.join(parameters["logging"]["model_configs_root"], name + "_configs.json")) as file:
        json.dump(obj=model.to_json(), fp=file)

    # Define callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=parameters["logging"]["weights_root"] + name + "_weights.hdf5",
        monitor=parameters["hyperparameters"]["monitor"],
        verbose=1,
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=parameters["hyperparameters"]["patience"],
                                                      min_delta=parameters["hyperparameters"]["min_delta"],
                                                      monitor=parameters["hyperparameters"]["monitor"])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=parameters["logging"]["tensorboard_root"] + name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training
    initialize_vars(sess=sess)
    model.fit([X_train[0], X_train[1], X_train[2]], y_train,
              validation_data=([X_valid[0], X_valid[1], X_valid[2]], y_valid),
              batch_size=parameters["hyperparameters"]["batch_size"],
              epochs=parameters["hyperparameters"]["epochs"],
              verbose=1,
              shuffle=parameters["hyperparameters"]["shuffle"],
              callbacks=[model_checkpoint, early_stopping, tensorboard_callback]
              )


def main(args):
    parameters = read_parameters(args.parameters_file)

    run_training(parameters=parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training with parameters predefined in a json file.")

    parser.add_argument("parameters_file",
                        metavar="p",
                        help="The json file containing the parameters.",
                        type=str
                        )

    args = parser.parse_args()

    main(args=args)

    print(read_parameters("train_parameters.json"))
