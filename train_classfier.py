#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

from jalef.preprocessing import BertPreprocessor, Word2VecPreprocessor
from jalef.layers import Bert
from jalef.models import Word2VecClassifier

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


def get_training_data(preprocessor, parameters):
    train = pd.read_csv(os.path.join(parameters["dataset"]["path"], "train.csv"))
    X_train, y_train = preprocessor.fit_transform_classification(train["Text"], train["Label"])

    valid = pd.read_csv(os.path.join(parameters["dataset"]["path"], "validation.csv"))
    X_valid, y_valid = preprocessor.fit_transform_classification(valid["Text"], valid["Label"])

    test = pd.read_csv(os.path.join(parameters["dataset"]["path"], "test.csv"))
    X_test, y_test = preprocessor.fit_transform_classification(test["Text"], test["Label"])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def run_training(parameters):
    name = "dataset=" + parameters["dataset"]["name"] + "_intents=" + str(parameters["dataset"]["n_intents"])

    # preprocessor = BertPreprocessor(max_sequence_length=parameters["dataset"]["max_seq_len"],
    #                                 pretrained_model_path=parameters["model"]["pretrained_model_path"])

    preprocessor = Word2VecPreprocessor(max_sequence_length=parameters["dataset"]["max_seq_len"])

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_training_data(preprocessor=preprocessor,
                                                                           parameters=parameters)

    # model = create_simple_bert_classifier(pretrained_model_path=parameters["model"]["pretrained_model_path"],
    #                                       output_size=parameters["model"]["output_size"],
    #                                       n_layers_to_finetune=parameters["model"]["n_layers_to_finetune"],
    #                                       max_seq_len=parameters["dataset"]["max_seq_len"],
    #                                       num_classes=len(y_train[0]))
    
    embedding_matrix = preprocessor.get_embedding_matrix(300, "models/word2vec/GoogleNews-vectors-negative300.bin.gz")

    model = Word2VecClassifier(n_classes=parameters["dataset"]["n_intents"],
                               time_steps=parameters["dataset"]["max_seq_len"],
                               fc_layer_sizes=[256, 128],
                               lstm_layer_sizes=[128],
                               name=name,
                               optimizer=parameters["hyperparameters"]["optimizer"],
                               loss=parameters["hyperparameters"]["loss"],
                               metrics=parameters["hyperparameters"]["metrics"],
                               monitor=parameters["hyperparameters"]["monitor"],
                               epochs=parameters["hyperparameters"]["epochs"],
                               batch_size=parameters["hyperparameters"]["batch_size"],
                               shuffle=parameters["hyperparameters"]["shuffle"],
                               patience=parameters["hyperparameters"]["patience"],
                               min_delta=parameters["hyperparameters"]["min_delta"],
                               weights_root=parameters["logging"]["weights_root"],
                               tensorboard_root=parameters["logging"]["tensorboard_root"]
                               )

    model.compile(print_summary=True, embedding_matrix=embedding_matrix)

    # Save model configuration
    with open(os.path.join(parameters["logging"]["model_configs_root"], name + "_configs.json"), "w") as file:
        json.dump(obj=model._model.to_json(), fp=file)

    initialize_vars(sess=sess)
    model.train(X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                load_best_model_on_end=True,
                evaluate_on_end=True,
                save_predictions_on_end=True,
                predictions_path=os.path.join("data/coursera_predictions/",
                                              "intents=" + str(parameters["dataset"]["n_intents"]) +
                                              "_predictions.npy"),
                verbose=1
                )

    # # Print layers
    # model.summary()
    #
    # # Compile model
    # model.compile(optimizer=parameters["hyperparameters"]["optimizer"],
    #               loss=parameters["hyperparameters"]["loss"],
    #               metrics=parameters["hyperparameters"]["metrics"])

    # # Define callbacks
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=parameters["logging"]["weights_root"] + name + "_weights.hdf5",
    #     monitor=parameters["hyperparameters"]["monitor"],
    #     verbose=1,
    #     save_best_only=True)
    #
    # early_stopping = tf.keras.callbacks.EarlyStopping(patience=parameters["hyperparameters"]["patience"],
    #                                                   min_delta=parameters["hyperparameters"]["min_delta"],
    #                                                   monitor=parameters["hyperparameters"]["monitor"])
    #
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=parameters["logging"]["tensorboard_root"] + name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    #
    # # Training
    # initialize_vars(sess=sess)
    # model.fit([X_train[0], X_train[1], X_train[2]], y_train,
    #           validation_data=([X_valid[0], X_valid[1], X_valid[2]], y_valid),
    #           batch_size=parameters["hyperparameters"]["batch_size"],
    #           epochs=parameters["hyperparameters"]["epochs"],
    #           verbose=1,
    #           shuffle=parameters["hyperparameters"]["shuffle"],
    #           callbacks=[model_checkpoint, early_stopping, tensorboard_callback]
    #           )


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
