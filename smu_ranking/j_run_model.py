# two functions one to train and validate and one to test the data
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from torchvision.datasets import ImageFolder

import gc
import time

def train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader):
    # If the model is DenseNet201, initialise a logger
    # Create a logger
    if (str(model) == "DenseNet201-abi"):
        logger = logging.getLogger('train_test_logger')

        # Create a file handler
        file_handler = logging.FileHandler(f'{model}_output.log')

        # Define the log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)  # Set the formatter for the file handler

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Set logging level
        logger.setLevel(logging.INFO)

    #total_step = len(j_load_data.train_loader)
    for epoch in range(num_epochs):
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            print(labels)

            # print("i am here 1")
            
            # Forward pass
            outputs = model(images)

            print(outputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("i am here 2")
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
            # print("i am here 3")

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, loss.item()))
        logger.info('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, loss.item()))
        
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(len(validation_loader), 100 * correct / total))
            print("Epoch time:",(time.time()-start), "s") 
            logger.info('Accuracy of the network on the {} validation images: {} %'.format(len(validation_loader), 100 * correct / total))
            logger.info("Epoch time:",(time.time()-start), "s")

def test(classes, device, model, test_loader):
    if (str(model) == "DenseNet201-abi"):
        logger = logging.getLogger('train_test_logger')

        # Create a file handler
        file_handler = logging.FileHandler(f'{model}_output.log')

        # Define the log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)  # Set the formatter for the file handler

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Set logging level
        logger.setLevel(logging.INFO)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))   
        logger.info('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

def test_indivclass(classes, test_loader, device, model):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    if (str(model) == "DenseNet201-abi"):
        logger = logging.getLogger('train_test_logger')

        # Create a file handler
        file_handler = logging.FileHandler(f'{model}_output.log')

        # Define the log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)  # Set the formatter for the file handler

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Set logging level
        logger.setLevel(logging.INFO)

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label >= len(classes):  # Check if label is out of range
                    print(f"Label {label} is out of range for classes tuple")
                    logger.info(f"Label {label} is out of range for classes tuple")
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        logger.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def create_test_res(img_dir, device, model):
    test_dataset = ImageFolder(root=img_dir)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    test_image_names = [item[0].split('/')[-1] for item in test_dataset.imgs]

    #create text file
    f = open("id_est.txt", "w")

    #do the predictions
    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #put the smu-like score inside 
            print(outputs.data)
            _, predict = torch.max(outputs.data, 1)
            res = zip(_, predict)
            for num, label in outputs.data:
                n = 0
                if label == 0: #change to index of not smu
                    n = 1 - num
                else:
                    n = num
                f.write(test_image_names[i] + " " + n +  '\n')
                i += 1
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            del images, labels, outputs
    
    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))
    f.close()
    