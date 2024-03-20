# two functions one to train and validate and one to test the data
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import gc
import time

def train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader, scheduler):
    #total_step = len(j_load_data.train_loader)
    for epoch in range(num_epochs):
        start = time.time()
        losses = []
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, loss.item()))
        
        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)


    
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

def test(device, model, test_loader):
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

def test_indivclass(classes, test_loader, device, model):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

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
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')    