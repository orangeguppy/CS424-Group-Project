import torch
import torch.nn as nn
from torchvision import models
from torch import optim
from torchvision.models import densenet201, DenseNet201_Weights


# Assuming these are the names of your modules
import j_load_data
import j_run_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        # Load a pre-trained densenet201 and replace the classifier
        #original_model = models.densenet201(pretrained=True)
        self.base_model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.classifier.in_features
        # self.model = nn.Sequential(
        #     original_model,
        #     nn.Linear(num_ftrs, num_classes)
        # )
        self.base_model.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

    def __str__(self):
        return "DenseNet201-abi"

if __name__ == "__main__":
    # Hyperparameters
    num_classes = 42  #change acc to num of folders in images
    num_epochs = 5
#if too small -- not enough -> no learning
#if too big - may start overfitting
    batch_size = 16
    learning_rate = 0.01

    # Model initialization
    model = DenseNet201(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    
    # Load data
    #train_loader, validation_loader, test_loader = j_load_data.create_data(batch_size=batch_size)
    train_loader, validation_loader, test_loader, classes = j_load_data.create_data()
    
    # Train and validate the model
    j_run_model.train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader)
    PATH = './densenet.pth'
    torch.save(model.state_dict(), PATH)
    
    # Test the model
    j_run_model.test(device, model, test_loader)
