# NDSC Product Matching Model 1 
Note : All script must be run in `shopee_ndsc/model_image_text_dian`

## Quickstart
1. Create a folder `shopee_ndsc/model_image_text_dian/data`. Copy all tfrecords data and raw data here. At least this folder should contain 2 folders and 2 files, folders `train` and `test`. Each folder contains tfrecords files. And the files used for generating tfrecords `train.csv` and `test.csv`. 
2. Create a folder `shopee_ndsc/model_image_text_dian/base_model`. This folder will be your `model_dir`.
3. Create `params.json` in the folder above. Put all model parameters here. Here is an example of `params.json`
```
{
    "title_model_type": "CNN",
    "title_max_length": 15,
    "title_w2v_dim": 256,
    "title_units": [64,64,64],
    "title_kernel_size": [1,2,3],
    "title_dropout_rate": 0.1,
    "desc_model_type": "CNN",
    "desc_max_length": 15,
    "desc_w2v_dim": 256,
    "desc_units": [64,64,64],
    "desc_kernel_size": [1,2,3],
    "desc_dropout_rate": 0.1,
    "image_model_type": "MobileNetV2",
    "image_size": 160,
    "image_dropout_rate": 0.1,
    "price_dimension": 1,
    "price_dropout_rate": 0.1,
    "selected_features": ["title", "desc", "image", "price"],
    "optimizer": "adam",
    "epoch": 10,
    "learning_rate":0.01,
    "batch_size":32,
    "global_dropout_rate":0.1
}
```
4. Create a folder `shopee_ndsc/model_image_text_dian/resources`. Put word2vec vectors here. There must be a file`title_vectors`. You can also put images and vocabularies here.

### Generate tfrecords
Generate tfrecords. Run these command.
```
create_tfrecords.py --input_dir data --output_dir data --resources_dir resources
```

### Train data
Train your model. Run these command.
```
python train.py --data_dir data --model_dir base_model --resources_dir resources
```
