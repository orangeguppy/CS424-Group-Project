# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

# Copyable code: imports
import j_load_data
import j_run_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M",128, 128,"M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }

#VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M',]
#then flatten and 4096x4096x1000 linear layers

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
    #def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        # fully connected layer
        self.fcs = nn.Sequential( 
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int: #if the type is an int --> we can know its a conv layer
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers) #unpacking all that we stored in the empty list


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = VGG_net(in_channels=3, num_classes=1000).to(device)
#     BATCH_SIZE = 3
#     x = torch.randn(3, 3, 224, 224).to(device)
#     assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
#     print(model(x).shape)

############################################################################################################
# Setting hyperparamters
num_classes = 5
num_epochs = 10
batch_size = 16
learning_rate = 0.01

#model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
model = VGG_net(in_channels=3, num_classes=5).to(device)

x = torch.randn(3, 3, 224, 224).to(device)

assert model(x).shape == torch.Size([batch_size, 5])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  


############################################################################################################
# Run model

# Copyable code to use the other two python files (Everything in between change)
# create the dataset
train_loader, validation_loader, test_loader = j_load_data.create_data()

# train the dataset
j_run_model.train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader)

# test the dataset
j_run_model.test(device, model, test_loader)
