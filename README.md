# Predict attributes from product image
Visual Taxonomy: Predict attribute values of products

## Overview
Have you ever encountered a product listing on an e-commerce platform where the image showed a short - sleeve shirt, but the description claimed it was long-sleeved? Such discrepancies are not just frustrating for customers, they're a significant challenge for e-commerce platforms striving to maintain accurate product catalogs at scale.

## Problem Statement
The competition challenges to develop a robust machine learning model that can accurately predict various product attributes (such as color, pattern, and sleeve length) solely from product images.

## Solution
Our solution focuses on accurately predicting various product attributes (such as color, pattern, and sleeve length) using a machine learning model trained on product images. Here’s an overview of the approach:

#### Data Preparation:

* The dataset consists of product images and corresponding labels for different attributes.
* We performed data cleaning and preprocessing, including handling missing values and normalizing the images.
* The images were augmented to increase the diversity of the training data and improve model generalization.

#### Model Architecture:

* We utilized a Convolutional Neural Network (CNN) architecture, such as EfficientNet or ResNet, for feature extraction from images.
* The final layers of the network were customized for multi-label classification to predict multiple attributes simultaneously.

#### Training Strategy:

* The model was trained using a combination of cross-entropy loss and other relevant metrics to optimize the attribute predictions.
* We employed techniques like multiple learning rate, dropout, batch normalisation and data shuffling to enhance the training process.

#### Evaluation and Testing:

* The model’s performance was evaluated using metrics like F1-score and accuracy.
* We used test set to tune hyperparameters and avoid overfitting.
* The best setting, which provide maximum F1-score till now is considered as default in our code.

#### Post-Processing:

* The model's outputs were refined using thresholding techniques to adjust predictions for each attribute.
* Additional checks were implemented to ensure consistency between predicted attributes.

### Step 1
```
pip install -r requirements.txt
```

### Step 2
Download the dataset from 
Make sure that the dataset is in `dataset` folder having `train.csv`, `test.csv`, `category_attributes.parquet`, `train_images`, and `test_images`.

### Step 3
```
cd ../src
```

### Step 4
```
python main.py
```
The above code only run in default setting (best for 16GB GPU) though the further attributes (hyperparameters and model) can be passed, including learning rate as `--lr`, maixmum pixel size `--pixel`, batch size `--batch_size`, model name `--model_name`, and path where model get saved as `--save_path`, as per requirements. For example
#### For running on EfficientNet by fine tuning it:
```
python main.py --lr 0.0001 --batch_size 32 --pixel 640 --model_name DeepMultiOutputModel_EfficientNet --save_path model
```
####  For running on RegNet by fine tuning it:
```
python main.py --lr 0.0001 --batch_size 32 --pixel 640 --model_name DeepMultiOutputModel_RegNet --save_path model
```
####  For running on our custom build model:
```
python main.py --lr 0.0001 --batch_size 1024 --pixel 640 --model_name DeepMultiOutputModel --save_path model
```
For our custom build model, we recommend using `batch_size` more than 512 to get better stability while training.

## Conclusion

Developed and optimized multi-output CNN models (including custom CNN, RegNet-16GF, and EfficientNet) to predict 10 different product attributes (e.g., color, pattern, sleeve length) from e-commerce product images, achieving an average F1 score of 0.6487.

