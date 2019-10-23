#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from jalef.models import BertClassifier
from jalef.preprocessing import BertPreprocessor
from jalef.callbacks import *

# set random seeds
np.random.seed(1234)
tf.set_random_seed(1234)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)


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


def get_training_data(preprocessor, parameters):
    train = pd.read_csv(os.path.join(parameters["dataset"]["path"], "train.csv"))
    X_train, y_train = preprocessor.fit_transform_classification(train["Text"], train["Label"])

    valid = pd.read_csv(os.path.join(parameters["dataset"]["path"], "validation.csv"))
    X_valid, y_valid = preprocessor.fit_transform_classification(valid["Text"], valid["Label"])

    test = pd.read_csv(os.path.join(parameters["dataset"]["path"], "test.csv"))
    X_test, y_test = preprocessor.fit_transform_classification(test["Text"], test["Label"])

    lut = pd.read_csv(os.path.join(parameters["dataset"]["path"], "lut.csv"), names=["Intent"], index_col=0, header=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, lut


def run_training(parameters):
    name = "dataset=" + parameters["dataset"]["name"] + "_embedding=" + parameters["model"]["embedding"] + \
           "_intents=" + str(parameters["dataset"]["n_intents"])

    print("Starting...")

    preprocessor = BertPreprocessor(pretrained_model_path=parameters["model"]["pretrained_model_path"],
                                    max_sequence_length=parameters["dataset"]["max_seq_len"])

    print("Loading dataset...")

    X_train, y_train, X_valid, y_valid, X_test, y_test, lut = get_training_data(preprocessor=preprocessor,
                                                                                parameters=parameters)

    print("Compiling model...")

    model = BertClassifier(
        pretrained_model_path=parameters["model"]["pretrained_model_path"],
        output_size=parameters["model"]["output_size"],
        n_layers_to_finetune=parameters["model"]["n_layers_to_finetune"],
        n_classes=parameters["dataset"]["n_intents"],
        time_steps=parameters["dataset"]["max_seq_len"],
        fc_layer_sizes=[256, 128],
        lstm_layer_sizes=[128],
        name=name,
        logs_root_path=parameters["logging"]["weights_root"],
    )

    model.compile(
        optimizer=parameters["hyperparameters"]["optimizer"],
        loss=parameters["hyperparameters"]["loss"],
        metrics=parameters["hyperparameters"]["metrics"],
        print_summary=True
    )

    svr = StoreValidationResults(x=X_valid,
                                 y=y_valid,
                                 lookup_fn=lambda e: lut.at[e, "Intent"]
                                 )

    model.add_custom_callback(svr)

    scm = SaveConfusionMatrix(svr_instance=svr,
                              title=name + "_epoch={}"
                              )

    model.add_custom_callback(scm)

    stp = SaveTestPredictions(
        X=X_test
    )

    model.add_custom_callback(stp)

    print("Start training...")

    initialize_vars(sess=sess)
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        epochs=parameters["hyperparameters"]["epochs"],
        monitor=parameters["hyperparameters"]["monitor"],
        patience=parameters["hyperparameters"]["patience"],
        min_delta=parameters["hyperparameters"]["min_delta"],
        verbose=1
    )

    print("Done.")


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
