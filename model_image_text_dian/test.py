import argparse
import joblib
import json
import logging
import mlflow
import os
import traceback

import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from input_pipeline import create_dataset
from mlflow import log_metrics, log_metric, log_param, log_artifact
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from model import build_tobacco_model

mlflow_server = "http://172.21.39.18:5000"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", 
    default="data/", 
    help="Directory containing the tfrecords dataset."
)
parser.add_argument(
    "--model_dir",
    help="Path to saved model / weights",
)
parser.add_argument(
    "--params",
    help="Path to params."
)
parser.add_argument(
    "--resources_dir",
    default="models/w2v",
    help="Directory containing word2vec"
)
parser.add_argument(
    "--log_mlflow",
    action="store_true",
    help="Log experiment to MLFLow, please be sure to connect to VPN"
)
parser.add_argument(
    "--credentials",
    required=True,
    help="Path to google credentials file, \
        go to https://sites.google.com/tokopedia.com/scientific-matters/dev-resource/using-mlflow?authuser=0 \
        for details"
)

def init_mlflow(google_authentication_file):
    mlflow.set_tracking_uri(mlflow_server)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_authentication_file
    
    return

if __name__ == "__main__":
    args = parser.parse_args()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.model_dir, subdir, "logs/")
    model_dir = os.path.join(args.model_dir, subdir, "models/")

    json_path = args.params
    assert os.path.isfile(json_path), "No json configuration found at {}".format(
        json_path
    )
    with open(json_path) as json_file:
        params = json.load(json_file)

    test_path = os.path.join(args.data_dir, "test/*")

    test_dataset = create_dataset(
        input_patterns=test_path,
        is_training=False,
        selected_features=["token_title", "token_desc", "image", "price", "product_id"],
        label="Tobacco"
    )

    title_weights_path = os.path.join(args.resources_dir, "title_vectors")
    desc_weights_path = os.path.join(args.resources_dir, "desc_vectors")
    title_weights = joblib.load(title_weights_path)
    desc_weights = joblib.load(desc_weights_path)

    model = build_tobacco_model(params=params, 
                                title_w2v_weights=title_weights, 
                                desc_w2v_weights=desc_weights
                                )
    model.load_weights(args.model_dir)
    model.save("best_model")
    history_test = model.evaluate(test_dataset)

    print("Predicting...")
    p_ids = []
    prediction = []
    for record in test_dataset:
        p_ids += list(record[0]['product_id'].numpy()[:,0])
        pred = model.predict(record[0])
        prediction += list(pred[:,0])
        print(prediction)
    
    result = {"product_id": p_ids, "confidence":prediction}
    result = pd.DataFrame(data=result)
    result.to_csv("prediction.csv")
    print("Prediction result exported!")

    init_mlflow(args.credentials)
    experiment_name = "tobacco_detection/combined_model"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.set_tags({'task': 'test', 'data': '20042020-test'})
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)

        # Log  parameters
        log_param("batch_size", str(params["batch_size"]))
        log_param("epochs", str(params["epoch"]))
        log_param("learning_rate", str(params["learning_rate"]))
        log_param("image_size", str(params["image_size"]))
        log_param("optimizer", str(params["optimizer"]))
        log_param("title_model_type", str(params["title_model_type"]))
        log_param("title_max_length", str(params["title_max_length"]))
        log_param("title_w2v_dim", str(params["title_w2v_dim"]))
        log_param("title_units", str(params["title_units"]))
        log_param("title_kernel_size", str(params["title_kernel_size"]))
        log_param("title_dropout_rate", str(params["title_dropout_rate"]))
        log_param("desc_model_type", str(params["desc_model_type"]))
        log_param("desc_max_length", str(params["desc_max_length"]))
        log_param("desc_w2v_dim", str(params["desc_w2v_dim"]))
        log_param("desc_units", str(params["desc_units"]))
        log_param("desc_kernel_size", str(params["desc_kernel_size"]))
        log_param("desc_dropout_rate", str(params["desc_dropout_rate"]))

        # Log test metrics
        test_loss = history_test[0]
        test_acc = history_test[1]
        test_recall = history_test[2]
        test_precision = history_test[3]

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_precision", test_precision)
