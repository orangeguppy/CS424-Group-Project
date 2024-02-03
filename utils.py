"""
Utility functions
"""
import logging
import subprocess

def generate_smu_logo_dataset():
    pass

def generate_dataset(dataset_dir):
    # Default transforms without augmentation
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (150, 150)
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),          # Convert images to PyTorch tensors
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
    dataset = ConcatDataset([default_dataset, augmented_dataset, augmented_dataset_2])
    return dataset

# This method should be called to train the model on the Training set. 
# Need to include code for saving the model weights when performance on the Validation set is best
def train(model, device, trainloader, num_epochs, batch_size, lr):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 60 == 59:    # print every 60 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 60:.3f}')
            running_loss = 0.0

    print('Finished Training')

# For testing model performance
def test(test_loader, model, device, PATH, loader_type="test"):
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0

    # Disable gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Log metrics
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the network on the {total} {loader_type} images: {accuracy} %')
    print(f'Accuracy of the network on the {total} {loader_type} images: {accuracy} %')

    return accuracy

def setup_logging():
    # Create a logger
    logger = logging.getLogger('train_test_logger')

    # Create a file handler
    file_handler = logging.FileHandler('output.log')

    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)  # Set the formatter for the file handler

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Set logging level
    logger.setLevel(logging.INFO)

    return logger