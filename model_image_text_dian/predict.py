import argparse
import io
import json
import requests
import sys
import urllib
import os

import re
import numpy as np
import tensorflow as tf

from PIL import Image

def main(args):
    model = tf.keras.models.load_model(args.model_path)

    title_vocab_path = os.path.join(args.resources_dir, "title_vocab.json")
    with open(title_vocab_path) as f:
        title_vocab = json.load(f)
    desc_vocab_path = os.path.join(args.resources_dir, "desc_vocab.json")
    with open(desc_vocab_path) as f:
        desc_vocab = json.load(f)

    if model.layers[0].name != 'image':
        model.layers[0]._name = 'image'

    # model.summary()

    inputs = get_inputs(args.product_id, title_vocab, desc_vocab)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(1)

    preds = model.predict(dataset)
    print(preds, type(preds))

def get_inputs(product_id, title_vocab, desc_vocab):
    endpoint = 'https://tome.tokopedia.com/v2.1/product/'
    inputs = {}
    try:
        resp = json.loads(urllib.request.urlopen(endpoint + str(product_id)).read().decode())
        inputs['token_title'] = preprocess_title(resp['data']['product_name'], title_vocab, 15) # need to preprocess title first; dummy example: np.expand_dims(np.zeros(15), axis=0)
        inputs['token_desc'] = preprocess_desc(resp['data']['product_description'], desc_vocab, 200) # need to preprocess description first; dummy example: np.expand_dims(np.zeros(200), axis=0)
        inputs['image'] = np.array([get_image_data(resp['data']['product_picture'][0]['url_original'])])
        inputs['price'] = np.array(get_price_feature(resp['data']['product_price']))
    except:
        inputs['token_title'] = None
        inputs['token_desc'] = None
        inputs['image'] = None
        inputs['price'] = None
        
    return inputs

def get_price_feature(price):
    return [1] if price >= 4000 and price <= 4100000 else [0]

def get_image_data(image_url):
    try:
        image_data = requests.get(image_url).content
    except:
        # get default image
        image = Image.new('RGB', (224, 224))
        image_byte = io.BytesIO()
        image.save(image_byte, format='jpeg')
        image_data = image_byte.getvalue()

    image_data = tf.io.decode_jpeg(image_data, channels=3)
    image_data = tf.image.convert_image_dtype(image_data, tf.float32)
    image_data = tf.image.resize(image_data, [224, 224])
    
    return image_data

def tokenize(string, vocab, max_length):
    string = string.split()
    tokens = []
    for i in string:
        try :
            tokens.append(vocab[i])
        except :
            tokens.append(len(vocab)+1)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        while len(tokens) < max_length:
            tokens.append(0)
    return np.array([tokens])

def parse_desc(product) :
    product = str(product)
    product = re.sub("_x000D_", " ", product)
    result_re = re.sub("[^A-Za-z0-9']+", " ", product)
    result_lower = result_re.lower()
    return result_lower

def preprocess_desc(product, vocab_list, max_length):
    product =  parse_desc(product)
    product =  tokenize(product, vocab_list, max_length)
    return product

def parse_title(product) :
    result_re = re.sub("[^A-Za-z0-9']+", ' ', product)
    result_lower = result_re.lower()
    return result_lower

def preprocess_title(product, vocab_list, max_length):
    product =  parse_title(product)
    product = tokenize(product, vocab_list, max_length)
    return product
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, 
        help='Path a directory containing the SavedModel')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--product_id', type=int,
        help='Product id.', default=777472946)
    parser.add_argument('--resources_dir', type=str,
        help='Path a directory containing title and description vocabs')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))