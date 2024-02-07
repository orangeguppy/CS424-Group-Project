import os
import torch
import model
# import utils
#import mlflow

from PIL import Image
#from pillow_heif import register_heif_opener
#register_heif_opener()

# This line is to set the model to use the GPU if available, else the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Setup logger
# logger = utils.setup_logging()

# Generate Dataset
# dataset = generate_smu_logo_dataset(dataset_downloaded=False)

# Specify hyperparameters, we probably will need to pass these hyperparameters into the train() method later on!
num_epochs = 3
batch_size = 16
lr = 0.001

# Split dataset and create Dataloaders for each training set
# Need to create the train, validation, and testing dataset
# To load the data from each of the train, validation, and testing datasets, we need to create corresponding Dataloaders.

# Instantiate model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device) # Move the model to the GPU if available, else move it to the CPU

# Directory containing images
#pandas.read_csv(r"C:\Users\DeePak\Desktop\myac.csv")
#change the below dir to the path in ur device :D
img_dir = r'C:\Users\Abiya\OneDrive\CS424-Group-Project\dataset\smu_logo\images'

# Get list of image files in directory
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

# Inference on each image
for img_path in img_files:
    # Load image
    image = Image.open(img_path)
    
    # Inference
    results = model(image)
    
    # Results
    print(f"Results for {img_path}:")
    results.print()
    results.save()  # or .show()

    print(results.xyxy[0])  # img predictions (tensor)
    print(results.pandas().xyxy[0])  # img predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

# # Train the model
# utils.train(model, device=device)

# # Test the model
# utils.test(model, device=device)