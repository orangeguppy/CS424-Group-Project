#  load the data from the dataset
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
import os
from collections import defaultdict


def create_data(batch_size):
    dataset_dir = "../dataset/smu_images"

    class_list = []
    items = os.listdir(dataset_dir)
    # Iterate through each item
    for item in items:
        # Check if the item is a folder
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # List all items in the subfolder
            sub_items = os.listdir(item_path)
            # Count how many of these items are files
            file_count = sum(1 for sub_item in sub_items if os.path.isfile(os.path.join(item_path, sub_item)))
            print(f"{item}: {file_count} files")
            class_list.append(item)
    # non_smu_index = class_list.index("not smu")
    # print(non_smu_index)
            
    # Default transforms without augmentation
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (150, 150)
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
    ])

    # Transforms to augment the dataset
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (150, 150)
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 0.5
        transforms.RandomRotation(degrees=15),  # Randomly rotate images by up to 15 degrees
        transforms.ToTensor(),        # Convert images to PyTorch tensors
        # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255.0, 1/255.0, 1/255.0])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Count images per label in the full_dataset
    label_counts = defaultdict(int)
    for _, label in default_dataset.samples:  # Using default_dataset to count, could use any
        label_counts[label] += 1

    # Print counts per label
    for label, count in label_counts.items():
        label_name = default_dataset.classes[label]
        print(f"Label {label} ({label_name}): {count} images")

    return class_list, train_loader, validation_loader, test_loader

