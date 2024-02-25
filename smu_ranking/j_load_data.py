#  load the data from the dataset
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
#from torchvision.transforms import transforms

#import j_load_data

def create_data():
    dataset_dir = "../dataset/smu_images"

    # Default transforms without augmentation
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (150, 150)
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255.0, 1/255.0, 1/255.0])
    ])

    # Transforms to augment the dataset
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (150, 150)
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 0.5
        transforms.RandomRotation(degrees=15),  # Randomly rotate images by up to 15 degrees
        transforms.ToTensor(),        # Convert images to PyTorch tensors
        # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255.0, 1/255.0, 1/255.0])
    ])

    default_dataset = ImageFolder(root=dataset_dir, transform=default_transform)
    augmented_dataset = ImageFolder(root=dataset_dir, transform=augment_transform)
    augmented_dataset_2 = ImageFolder(root=dataset_dir, transform=augment_transform)
    full_dataset = ConcatDataset([default_dataset, augmented_dataset, augmented_dataset_2])
        
    #split into 3 datasets
    train_size = int(0.7 * len(full_dataset))  
    validation_size = int(0.15 * len(full_dataset))  
    test_size = len(full_dataset) - train_size - validation_size  
    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])


    #create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    return train_loader, validation_loader, test_loader

