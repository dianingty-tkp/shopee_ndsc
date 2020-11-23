import logging
import argparse
import os
import json
import traceback

import mlflow
import mlflow.keras
import mlflow.tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from datetime import datetime
from mlflow import log_metrics, log_metric, log_param, log_artifact
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from model import build_model
from input_pipeline import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", 
    default="data/", 
    help="Directory containing the tfrecords dataset."
)
parser.add_argument(
    "--model_dir",
    default="models/base_model",
    help="Directory containing params.json",
)
parser.add_argument(
    "--resources_dir",
    default="models/w2v",
    help="Directory containing word2vec"
)
    
#     return

if __name__ == "__main__":
    args = parser.parse_args()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.model_dir, subdir, "logs/")
    model_dir = os.path.join(args.model_dir, subdir, "models/")

    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration found at {}".format(
        json_path
    )
    with open(json_path) as json_file:
        params = json.load(json_file)

    train_path = os.path.join(args.data_dir, "train/*")
    test_path = os.path.join(args.data_dir, "test/*")

    train_dataset = create_dataset(
        input_patterns=train_path,
        selected_features=["token_title_1", "token_title_2", "byte_image_1", "byte_image_2"],
        label="Label"
    )
    test_dataset = create_dataset(
        input_patterns=test_path,
        selected_features=["token_title_1", "token_title_2", "byte_image_1", "byte_image_2"],
        label="Label"
    )

    title_weights_path = os.path.join(args.resources_dir, "title_vectors")
    title_weights = joblib.load(title_weights_path) # maybe we need to chane this one

    model = build_model(
        params=params, 
        title_w2v_weights=title_weights)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
        )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights", "best_weights.ckpt"), 
        save_best_only=True, 
        save_weights_only=True
        )
    
    history = model.fit(
        train_dataset, 
        epochs=params["epoch"], 
        verbose=1, 
        validation_data=test_dataset, 
        callbacks=[tensorboard, checkpoint]
        )
    
    last_model_path = os.path.join(model_dir, "last_model/")
    model.save(last_model_path, "tf")