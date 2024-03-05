# def VGGupdated(input_tensor=None,classes=2):    
   
#     img_rows, img_cols = 300, 300   # by default size is 224,224
#     img_channels = 3

#     img_dim = (img_rows, img_cols, img_channels)
   
#     img_input = Input(shape=img_dim)
    
#     # Block 1
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#     # Block 2
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

#     # Block 3
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#     # Block 4
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

#     # Block 5
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
#     # Classification block
#     x = Flatten(name='flatten')(x)
#     x = Dense(4096, activation='relu', name='fc1')(x)
#     x = Dense(4096, activation='relu', name='fc2')(x)
#     x = Dense(classes, activation='softmax', name='predictions')(x)

#     # Create model.
   
     
#     model = Model(inputs = img_input, outputs = x, name='VGGdemo')


#     return model

# Imports
# import torch
# import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

# VGG_types = {
#     "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "VGG16": [
#         64,
#         64,
#         "M",
#         128,
#         128,
#         "M",
#         256,
#         256,
#         256,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#     ],
#     "VGG19": [
#         64,
#         64,
#         "M",
#         128,
#         128,
#         "M",
#         256,
#         256,
#         256,
#         256,
#         "M",
#         512,
#         512,
#         512,
#         512,
#         "M",
#         512,
#         512,
#         512,
#         512,
#         "M",
#     ],
# }

# #VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M',]
# #then flatten and 4096x4096x1000 linear layers

# class VGG_net(nn.Module):
#     def __init__(self, in_channels=3, num_classes=1000):
#     #def __init__(self, in_channels, num_classes):
#         super(VGG_net, self).__init__()
#         self.in_channels = in_channels
#         self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

#         # fully connected layer
#         self.fcs = nn.Sequential( 
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fcs(x)
#         return x

#     def create_conv_layers(self, architecture):
#         layers = []
#         in_channels = self.in_channels

#         for x in architecture:
#             if type(x) == int: #if the type is an int --> we can know its a conv layer
#                 out_channels = x

#                 layers += [
#                     nn.Conv2d(
#                         in_channels=in_channels,
#                         out_channels=out_channels,
#                         kernel_size=(3, 3),
#                         stride=(1, 1),
#                         padding=(1, 1),
#                     ),
#                     nn.BatchNorm2d(x),
#                     nn.ReLU(),
#                 ]
#                 in_channels = x
#             elif x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

#         return nn.Sequential(*layers) #unpacking all that we stored in the empty list


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = VGG_net(in_channels=3, num_classes=1000).to(device)
#     BATCH_SIZE = 3
#     x = torch.randn(3, 3, 224, 224).to(device)
#     assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
#     print(model(x).shape)
    
# model = VGG_net(in_channels=3, num_classes=1000)
# x = torch.randn(1, 3, 224, 224)
# print(model(x).shape)