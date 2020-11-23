import argparse
import logging
import os
import time
import timeit
import json

import pandas as pd
import tensorflow as tf
import collections

from tfrecord.text_utils import preprocess_text
from tfrecord.image_utils import preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, 
                    help='Folder path to input directory', required=True, default=None)
parser.add_argument('--output_dir', type=str, 
                    help='Folder path to output directory', required=True, default=None)
parser.add_argument('--resources_dir', type=str, 
                    help='Directory containing resources (image and vocabulary)', required=True, default=None)

def write_tfrecords(dataframe, output_dir, num_files=1000):
    writers = []
    for i in range(num_files):
        file_name = "tfrecords_" + str(i)
        file_path = os.path.join(output_dir, file_name)
        writers.append(tf.io.TFRecordWriter(file_path))

    dict_ = dataframe.to_dict()
    features = collections.OrderedDict()

    for i in range(len(dataframe)):
        if (i % 100 == 0):
            print(str(i), " records done!")

        features["token_title_1"]=tf.train.Feature(int64_list=
                                            tf.train.Int64List(
                                                value=dict_["token_title_1"][i]))
        features["token_title_2"]=tf.train.Feature(int64_list=
                                            tf.train.Int64List(
                                                value=dict_["token_title_2"][i]))
        features["byte_image_1"]=tf.train.Feature(float_list=
                                            tf.train.FloatList(
                                                value=dict_["byte_image_1"][i]))
        features["byte_image_2"]=tf.train.Feature(float_list=
                                            tf.train.FloatList(
                                                value=dict_["byte_image_2"][i]))
        features["Label"]=tf.train.Feature(int64_list=
                                             tf.train.Int64List(
                                                 value=[dict_["Label"][i]]))
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer_index = i % num_files
        writers[writer_index].write(tf_example.SerializeToString()) 

if __name__ == "__main__":
    args = parser.parse_args()

    title_vocab_path = os.path.join(args.resources_dir, "title_vocab.json")
    with open(title_vocab_path) as f:
        title_vocab = json.load(f)

    print("Importing data...")
    train_path = os.path.join(args.input_dir, "train.csv")
    test_path = os.path.join(args.input_dir, "test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("Preprocessing data...")
    print("Preprocessing Text: Train")
    train_data = preprocess_text(
        dataframe=train_data,
        desc_vocab=desc_vocab,
        desc_re_keywords=desc_re_keywords,
        title_vocab=title_vocab)
    print("Preprocessing Image: Train")
    train_data = preprocess_image(
        dataframe=train_data)
    print("Preprocessing Text: Test")
    test_data = preprocess_text(
        dataframe=test_data,
        desc_vocab=desc_vocab,
        desc_re_keywords=desc_re_keywords,
        title_vocab=title_vocab)
    print("Preprocessing Image: Test")
    test_data = preprocess_image(
        dataframe=test_data)

    print("Creating tfrecords for train data...")
    train_output_dir = os.path.join(args.output_dir, "train")
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    write_tfrecords(train_data, train_output_dir)

    print("Creating tfrecords for test data...")
    test_output_dir = os.path.join(args.output_dir, "test")
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    write_tfrecords(test_data, test_output_dir)

    print('Done!')