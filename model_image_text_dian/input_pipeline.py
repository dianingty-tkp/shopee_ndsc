"""Create input pipeline and return `tf.data` for the model.
Please refer to this code
https://github.com/tokopedia/data-science/blob/language_model/bert/research/bert_language_model/nlp/bert/input_pipeline.py
"""

import tensorflow as tf

def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
        if 'image' in name:
            image = tf.image.decode_jpeg(example[name], channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            example[name] = tf.image.resize(image, [224, 224])
        else:
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

    return example

def create_dataset(
    input_patterns,
    is_training=True,
    batch_size=32,
    n_buffer=128,
    n_prefetch=tf.data.experimental.AUTOTUNE,
    selected_features=None,
    label=None,
):
    """Creates input dataset from tfrecords files for training.
    Args:
        - input_patterns: a file pattern (or list of file patterns) to the tfrecords dataset
                The ideal input would be [dataset_file_pattern, feature_1_pattern, ...,
                                        feature_n_pattern]
        - is_training: whether the dataset is train dataset (default to True)
        - batch_size: the size of dataset batch (default to 32)
        - n_buffer: the size of buffer for shuffle (default to 128); it should be higher than
                    batch_size.
        - n_prefetch: the size of prefetch for data reading parallelization (default to 1024)
        - selected_features: the list of features to be used in the training process.
        - label: the label name (column name) from the data.
    Returns:
        - tf.data.Dataset object
    """

    if selected_features is None:
        raise ValueError("Please specify the features you want to use.")
    if label is None:
        raise ValueError("Please specify the label.")

    def _reshaper(record, key, shape):
        if key in record:
            record[key] = tf.reshape(record[key], shape)
        return record
    
    def _reshape_image(record, key, shape):
        record[key] = tf.image.resize(tf.image.convert_image_dtype(tf.io.decode_raw(record[key], tf.float32), tf.float32), shape)
        return record

    def _decode_fn(record):
        name_to_features = {
            "token_title_1": tf.io.FixedLenFeature([15], tf.int64),
            "token_title_2": tf.io.FixedLenFeature([15], tf.int64),
            "byte_image_1": tf.io.FixedLenFeature([], tf.string),
            "byte_image_1": tf.io.FixedLenFeature([], tf.string),
            "Label": tf.io.FixedLenFeature([1], tf.int64),
        }

        selected_name_to_features = {
            k: name_to_features[k] for k in selected_features + [label]}
        
        record = decode_record(record, selected_name_to_features)

        return record

    def _select_features(record):
        """Filter features to use for training."""
        features = {}
        for k in selected_features:
            features[k] = record[k]
        labels = record[label]
        return features, labels

    # list files from input patterns
    dataset = tf.data.Dataset.list_files(input_patterns)

    # set shuffle buffer to exactly match total number of training files to ensure that
    # training data is well shuffled.
    input_files = tf.io.gfile.glob(input_patterns)
    dataset = dataset.shuffle(len(input_files))

    # read and decode TFRecord
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(
        _decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(
        _select_features, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if is_training:
        dataset = dataset.shuffle(n_buffer)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(n_prefetch)

    return dataset

# if __name__ == '__main__':
#     path = '/data_2/dataset/tfrecord/tobacco/combined/train/'
#     input_patterns = 'tfrecords_*'
#     train_data = create_dataset(path + input_patterns, is_training=True, selected_features=["token_desc", "token_title", "image"], label="Tobacco")
#     print(type(train_data))
#     print(train_data)
#     for dic, label in train_data.take(1):
#         print(dic['token_desc'], dic['token_title'], dic['image'], label.numpy())
