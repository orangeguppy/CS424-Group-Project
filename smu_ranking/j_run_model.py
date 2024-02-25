import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import j_load_data
import j_rnn

import gc
total_step = len(j_load_data.train_loader)

#train
for epoch in range(j_rnn.num_epochs):
    for i, (images, labels) in enumerate(j_load_data.train_loader):  
        # Move tensors to the configured device
        images = images.to(j_rnn.device)
        labels = labels.to(j_rnn.device)
        
        # Forward pass
        outputs = j_rnn.model(images)
        loss = j_rnn.criterion(outputs, labels)
        
        # Backward and optimize
        j_rnn.optimizer.zero_grad()
        loss.backward()
        j_rnn.optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, j_rnn.num_epochs, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in j_load_data.validation_loader:
            images = images.to(j_rnn.device)
            labels = labels.to(j_rnn.device)
            outputs = j_rnn.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 

#testing
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in j_load_data.test_loader:
        images = images.to(j_rnn.device)
        labels = labels.to(j_rnn.device)
        outputs = j_rnn.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   