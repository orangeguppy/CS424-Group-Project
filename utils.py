"""
Utility functions for dataset/dataloader, training/testing/validation functions, and any other miscellaneous helper functions.
Add as needed!!
"""
import logging

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
def train(model, device):
    pass

# This method should be called for testing on both Validation and Test sets
# Make sure to include code to load the correct model weights
def test(model, device): 
    pass

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