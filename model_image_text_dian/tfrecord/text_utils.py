import pandas as pd
import json
import numpy as np
import re

from tqdm import tqdm

tqdm.pandas()

def tokenize(string, vocab, max_length):
    string = string.split()
    string = [i for i in string if i != ""]
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
    return np.array(tokens)

def parse_title(product) :
    result_re = re.sub("[^A-Za-z0-9']+", ' ', product)
    result_lower = result_re.lower()
    return result_lower

def preprocess_title(product, vocab_list, max_length):
    product =  product.progress_map(lambda product : parse_title(product))
    product = product.progress_map(lambda i : (tokenize(i, vocab_list, max_length)))
    return product

def preprocess_text(dataframe,
                    title_vocab,
                    title_max_length=15
                   ):

    dataframe["token_title_1"] = preprocess_title(
        dataframe["title_1"], 
        vocab_list=title_vocab, 
        max_length=title_max_length)
    
    dataframe["token_title_2"] = preprocess_title(
        dataframe["title_2"], 
        vocab_list=title_vocab, 
        max_length=title_max_length)
    
    return dataframe