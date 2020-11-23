from PIL import Image

import pandas as pd
import io

from tqdm import tqdm

tqdm.pandas()

def read_image(path):
    image = Image.open(path)
    return image

def resize_image(image, height=160, width=160):
    return image.resize((height, width))

def _to_bytes(path):
    image = read_image(path)

    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr 

def preprocess_image(dataframe):
    dataframe["byte_image_1"] = dataframe["image_1"].progress_map(lambda x: _to_bytes(x))
    dataframe["byte_image_2"] = dataframe["image_2"].progress_map(lambda x: _to_bytes(x))
    
    return dataframe