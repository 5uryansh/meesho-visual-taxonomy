from utils import FashionDataset
from utils import train_model
from utils import load_and_split_data

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# parsing
parser = argparse.ArgumentParser(description="Meesho")
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--pixel', type=int, default=640)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--model_name', type=str, default='DeepMultiOutputModel_EfficientNet')
parser.add_argument('--save_path', type=str, default='model')

args = parser.parse_args()
lr = args.lr
pixel = args.pixel
batch_size = args.batch_size
model_name = args.model_name
save_path = args.save_path
save_path = f'{save_path}.pth'


if model_name=='DeepMultiOutputModel_EfficientNet':
    from utils import DeepMultiOutputModel_EfficientNet
    
    # preprocessing the dataset
    train_df = pd.read_csv('../dataset/train.csv')
    train_df.fillna("dummy_value", inplace=True)
    
    mapping = {}
    for i in range(1,11):
        col = "attr_"+str(i)
        mapping[col] = train_df[col].unique()
        mapping[col] = mapping[col][mapping[col]!="dummy_value"]
        
        mapping[col]= np.insert(mapping[col],0,"dummy_value")
        
    Map = {}
    for i in range(0, len(mapping)):
        att = "attr_"+str(i+1)
        mpp = {}
        j = 0
        for attribute in mapping[att]:
            mpp[attribute]=j
            j+=1
        Map[att] = mpp

    train = train_df
    
    for at in Map.keys():
        mpp = Map[at]
        train[at] = train[at].map(mpp)
        
    train = train.drop(['Category','len'], axis=1)
    
    # locating the datasets
    train.to_csv("../dataset/pre_processed.csv")
    image_dir = '../dataset/train_images'
    csv_file = '../dataset/pre_processed.csv'
    save_path=save_path
    
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((int(0.75*pixel), pixel)),
        transforms.ToTensor(),
    ])
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMultiOutputModel_EfficientNet().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # initialising training
    train_model(model, csv_file=csv_file, image_dir=image_dir, transform=transform, optimizer=optimizer, criterion=criterion, device=device, batch_size=batch_size, num_epochs=20, save_path=save_path)

elif model_name=='DeepMultiOutputModel_RegNet':
    from utils import DeepMultiOutputModel_RegNet
    
    # preprocessing the dataset
    train_df = pd.read_csv('../dataset/train.csv')
    train_df.fillna("dummy_value", inplace=True)
    
    mapping = {}
    for i in range(1,11):
        col = "attr_"+str(i)
        mapping[col] = train_df[col].unique()
        mapping[col] = mapping[col][mapping[col]!="dummy_value"]
        
        mapping[col]= np.insert(mapping[col],0,"dummy_value")
        
    Map = {}
    for i in range(0, len(mapping)):
        att = "attr_"+str(i+1)
        mpp = {}
        j = 0
        for attribute in mapping[att]:
            mpp[attribute]=j
            j+=1
        Map[att] = mpp

    train = train_df
    
    for at in Map.keys():
        mpp = Map[at]
        train[at] = train[at].map(mpp)
        
    train = train.drop(['Category','len'], axis=1)
    
    # locating the datasets
    train.to_csv("../dataset/pre_processed.csv")
    image_dir = '../dataset/train_images'
    csv_file = '../dataset/pre_processed.csv'
    save_path=save_path
    
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((int(0.75*pixel), pixel)),
        transforms.ToTensor(),
    ])
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMultiOutputModel_RegNet().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # initialising training
    train_model(model, csv_file=csv_file, image_dir=image_dir, transform=transform, optimizer=optimizer, criterion=criterion, device=device, batch_size=batch_size, num_epochs=20, save_path=save_path)

elif model_name=='DeepMultiOutputModel':
    from utils import DeepMultiOutputModel
    
    # preprocessing the dataset
    train_df = pd.read_csv('../dataset/train.csv')
    train_df.fillna("dummy_value", inplace=True)
    
    mapping = {}
    for i in range(1,11):
        col = "attr_"+str(i)
        mapping[col] = train_df[col].unique()
        mapping[col] = mapping[col][mapping[col]!="dummy_value"]
        
        mapping[col]= np.insert(mapping[col],0,"dummy_value")
        
    Map = {}
    for i in range(0, len(mapping)):
        att = "attr_"+str(i+1)
        mpp = {}
        j = 0
        for attribute in mapping[att]:
            mpp[attribute]=j
            j+=1
        Map[att] = mpp

    train = train_df
    
    for at in Map.keys():
        mpp = Map[at]
        train[at] = train[at].map(mpp)
        
    train = train.drop(['Category','len'], axis=1)
    
    # locating the datasets
    train.to_csv("../dataset/pre_processed.csv")
    image_dir = '../dataset/train_images'
    csv_file = '../dataset/pre_processed.csv'
    save_path=save_path
    
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((int(0.75*pixel), pixel)),
        transforms.ToTensor(),
    ])
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMultiOutputModel().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # initialising training
    train_model(model, csv_file=csv_file, image_dir=image_dir, transform=transform, optimizer=optimizer, criterion=criterion, device=device, batch_size=batch_size, num_epochs=20, save_path=save_path)