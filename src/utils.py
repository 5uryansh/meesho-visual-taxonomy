import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision import models
from colorama import Fore, Style
from torchvision.models import regnet_y_16gf


class FashionDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row['id']).zfill(6)
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Extracting attributes
        attributes = row[['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5',
                          'attr_6', 'attr_7', 'attr_8', 'attr_9', 'attr_10']].values
        
        # Converting to tensor
        attributes = torch.tensor(attributes, dtype=torch.long)
        
        return image, attributes


# Training loop with F1 score calculation for multi-class classification
def train_model(model, csv_file, image_dir, transform, optimizer, criterion, device, 
                batch_size=32, num_epochs=20, save_path="model.pth"):
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        # Load and split the data for each epoch to shuffle train/test rows
        train_df, test_df = load_and_split_data(csv_file, test_size=0.2)
        
        # Preparing the dataloaders
        train_dataset = FashionDataset(data=train_df, img_dir=image_dir, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = FashionDataset(data=test_df, img_dir=image_dir, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        
        # Training Phase
        # Setting the model to training mode
        model.train()  
        running_loss = 0.0
        all_true_labels_train = [[] for _ in range(10)]
        all_predicted_labels_train = [[] for _ in range(10)]
        
        for images, attributes in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images = images.to(device)
            attributes = attributes.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Loss calculation for each attribute output
            loss = 0
            for i in range(10):
                loss += criterion(outputs[i], attributes[:, i])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Collecting predictions and true labels for F1 score calculation
            for i in range(10):
                _, predicted = torch.max(outputs[i], 1)
                all_true_labels_train[i].extend(attributes[:, i].cpu().numpy())
                all_predicted_labels_train[i].extend(predicted.cpu().numpy())
            

        # Average loss for the epoch
        avg_train_loss = running_loss / len(train_dataloader)
        
        # F1 scores for training set
        train_f1_scores = []
        for i in range(10):
            f1_macro_train = f1_score(all_true_labels_train[i], all_predicted_labels_train[i], average='macro')
            f1_micro_train = f1_score(all_true_labels_train[i], all_predicted_labels_train[i], average='micro')
            
            # Harmonic mean of f1_macro and f1_micro
            if (f1_macro_train + f1_micro_train) == 0:
                f1_train = 0
            else:
                f1_train = 2 * (f1_macro_train * f1_micro_train) / (f1_macro_train + f1_micro_train)
            
            train_f1_scores.append(f1_train)
        avg_train_f1_score = sum(train_f1_scores) / len(train_f1_scores)
        
        print(Fore.GREEN + f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Avg Training F1 Score: {avg_train_f1_score:.4f}" + Style.RESET_ALL)

        # Testing Phase
        model.eval()
        running_test_loss = 0.0
        all_true_labels_test = [[] for _ in range(10)]
        all_predicted_labels_test = [[] for _ in range(10)]
        
        with torch.no_grad():
            for images, attributes in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Testing"):
                images = images.to(device)
                attributes = attributes.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Loss calculation for each attribute output
                test_loss = 0
                for i in range(10):
                    test_loss += criterion(outputs[i], attributes[:, i])
                
                running_test_loss += test_loss.item()
                
                # True labels for F1 score calculation
                for i in range(10):
                    _, predicted = torch.max(outputs[i], 1)
                    all_true_labels_test[i].extend(attributes[:, i].cpu().numpy())
                    all_predicted_labels_test[i].extend(predicted.cpu().numpy())
        
        # Average test loss for the epoch
        avg_test_loss = running_test_loss / len(test_dataloader)
        
        # F1 scores for testing set
        test_f1_scores = []
        for i in range(10):
            f1_macro_test = f1_score(all_true_labels_test[i], all_predicted_labels_test[i], average='macro')
            f1_micro_test = f1_score(all_true_labels_test[i], all_predicted_labels_test[i], average='micro')
            
            # Harmonic mean of f1_macro and f1_micro
            if (f1_macro_test + f1_micro_test) == 0:
                f1_test = 0
            else:
                f1_test = 2 * (f1_macro_test * f1_micro_test) / (f1_macro_test + f1_micro_test)
            
            test_f1_scores.append(f1_test)
        avg_test_f1_score = sum(test_f1_scores) / len(test_f1_scores)
        
        print(Fore.RED + f"Epoch [{epoch+1}/{num_epochs}], Testing Loss: {avg_test_loss:.4f}, Avg Testing F1 Score: {avg_test_f1_score:.4f}" + Style.RESET_ALL)

        # Save the model checkpoint at the end of each epoch
        torch.save(model.state_dict(), save_path)
        print("Model saved to ", save_path)
        
        
# Function to load and split data into training and testing sets
def load_and_split_data(csv_file, test_size=0.2, random_state=None):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


# Efficient Net
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        efficientnet_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        self.features = efficientnet_b7.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

# Deep Multi-output Model for Attribute Prediction
class DeepMultiOutputModel_EfficientNet(nn.Module):
    def __init__(self):
        super(DeepMultiOutputModel_EfficientNet, self).__init__()
        self.feature_extractor = EfficientNetFeatureExtractor()
        self.fc_attr_1 = nn.Linear(2560, 19)
        self.fc_attr_2 = nn.Linear(2560, 15)
        self.fc_attr_3 = nn.Linear(2560, 11)
        self.fc_attr_4 = nn.Linear(2560, 20)
        self.fc_attr_5 = nn.Linear(2560, 15)
        self.fc_attr_6 = nn.Linear(2560, 8)
        self.fc_attr_7 = nn.Linear(2560, 12)
        self.fc_attr_8 = nn.Linear(2560, 11)
        self.fc_attr_9 = nn.Linear(2560, 14)
        self.fc_attr_10 = nn.Linear(2560, 9)

    def forward(self, x):
        # Extract features using EfficientNet
        features = self.feature_extractor(x)
        
        # Predict each attribute using the corresponding fully connected layer
        attr_1_output = F.softmax(self.fc_attr_1(features), dim=1)
        attr_2_output = F.softmax(self.fc_attr_2(features), dim=1)
        attr_3_output = F.softmax(self.fc_attr_3(features), dim=1)
        attr_4_output = F.softmax(self.fc_attr_4(features), dim=1)
        attr_5_output = F.softmax(self.fc_attr_5(features), dim=1)
        attr_6_output = F.softmax(self.fc_attr_6(features), dim=1)
        attr_7_output = F.softmax(self.fc_attr_7(features), dim=1)
        attr_8_output = F.softmax(self.fc_attr_8(features), dim=1)
        attr_9_output = F.softmax(self.fc_attr_9(features), dim=1)
        attr_10_output = F.softmax(self.fc_attr_10(features), dim=1)
        
        return (attr_1_output, attr_2_output, attr_3_output, attr_4_output, 
                attr_5_output, attr_6_output, attr_7_output, attr_8_output, 
                attr_9_output, attr_10_output)

# RegNetY-16GF
class RegNetY16GFFeatureExtractor(nn.Module):
    def __init__(self):
        super(RegNetY16GFFeatureExtractor, self).__init__()
        self.regnet_y_16gf = regnet_y_16gf(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(*list(self.regnet_y_16gf.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

# Deep Multi-output Model for Attribute Prediction
class DeepMultiOutputModel_RegNet(nn.Module):
    def __init__(self):
        super(DeepMultiOutputModel_RegNet, self).__init__()
        self.feature_extractor = RegNetY16GFFeatureExtractor()
        self.fc_attr_1 = nn.Linear(3024, 19)  
        self.fc_attr_2 = nn.Linear(3024, 15)  
        self.fc_attr_3 = nn.Linear(3024, 11)  
        self.fc_attr_4 = nn.Linear(3024, 20)  
        self.fc_attr_5 = nn.Linear(3024, 15)  
        self.fc_attr_6 = nn.Linear(3024, 8)   
        self.fc_attr_7 = nn.Linear(3024, 12)  
        self.fc_attr_8 = nn.Linear(3024, 11)  
        self.fc_attr_9 = nn.Linear(3024, 14)  
        self.fc_attr_10 = nn.Linear(3024, 9)  

    def forward(self, x):
        # Extract features using RegNetY-16GF
        features = self.feature_extractor(x)
        
        # Predict each attribute using the corresponding fully connected layer
        attr_1_output = F.softmax(self.fc_attr_1(features), dim=1)
        attr_2_output = F.softmax(self.fc_attr_2(features), dim=1)
        attr_3_output = F.softmax(self.fc_attr_3(features), dim=1)
        attr_4_output = F.softmax(self.fc_attr_4(features), dim=1)
        attr_5_output = F.softmax(self.fc_attr_5(features), dim=1)
        attr_6_output = F.softmax(self.fc_attr_6(features), dim=1)
        attr_7_output = F.softmax(self.fc_attr_7(features), dim=1)
        attr_8_output = F.softmax(self.fc_attr_8(features), dim=1)
        attr_9_output = F.softmax(self.fc_attr_9(features), dim=1)
        attr_10_output = F.softmax(self.fc_attr_10(features), dim=1)
        
        return (attr_1_output, attr_2_output, attr_3_output, attr_4_output, 
                attr_5_output, attr_6_output, attr_7_output, attr_8_output, 
                attr_9_output, attr_10_output)
    

# Deep CNN Feature Extractor Model from scratch
class DeepCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(DeepCNNFeatureExtractor, self).__init__()
        
        # First block of convolution + pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64
        
        # Second block of convolution + pooling
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32

        # Third block of convolution + pooling
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16
        
        # Fully connected layer after flattening
        self.fc1 = nn.Linear(1024 * 15 * 20, 512)
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(512, 128)  

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        
        # print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

# Deep Multi-output Model for Attribute Prediction
class DeepMultiOutputModel(nn.Module):
    def __init__(self):
        super(DeepMultiOutputModel, self).__init__()
        self.feature_extractor = DeepCNNFeatureExtractor()
        
        # Define output heads for 10 different attributes
        self.fc_attr_1 = nn.Linear(128, 19)
        self.fc_attr_2 = nn.Linear(128, 15)
        self.fc_attr_3 = nn.Linear(128, 11)
        self.fc_attr_4 = nn.Linear(128, 20)
        self.fc_attr_5 = nn.Linear(128, 15)
        self.fc_attr_6 = nn.Linear(128, 8)
        self.fc_attr_7 = nn.Linear(128, 12)
        self.fc_attr_8 = nn.Linear(128, 11)
        self.fc_attr_9 = nn.Linear(128, 14)
        self.fc_attr_10 = nn.Linear(128, 9)

    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Use extracted features to predict each attribute
        attr_1_output = F.softmax(self.fc_attr_1(features), dim=1)
        attr_2_output = F.softmax(self.fc_attr_2(features), dim=1)
        attr_3_output = F.softmax(self.fc_attr_3(features), dim=1)
        attr_4_output = F.softmax(self.fc_attr_4(features), dim=1)
        attr_5_output = F.softmax(self.fc_attr_5(features), dim=1)
        attr_6_output = F.softmax(self.fc_attr_6(features), dim=1)
        attr_7_output = F.softmax(self.fc_attr_7(features), dim=1)
        attr_8_output = F.softmax(self.fc_attr_8(features), dim=1)
        attr_9_output = F.softmax(self.fc_attr_9(features), dim=1)
        attr_10_output = F.softmax(self.fc_attr_10(features), dim=1)
        
        return (attr_1_output, attr_2_output, attr_3_output, attr_4_output, 
                attr_5_output, attr_6_output, attr_7_output, attr_8_output, 
                attr_9_output, attr_10_output)