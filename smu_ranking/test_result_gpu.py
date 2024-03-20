import torch
import os
#step1: comment out model not being tested
#import densenet201
import densenet121
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def merge_accuracies(logos_file, id_est_file, output_file):
    # Read data from the logo detection
    with open(logos_file, 'r') as logos:
        logos_data = logos.readlines()

    # Read data from desnet
    with open(id_est_file, 'r') as id_est:
        id_est_data = id_est.readlines()

    # Create a dictionary to store accuracies for each image
    accuracy_dict = {}

    # Populate accuracy dictionary from logos.txt
    for line in logos_data:
        filename, accuracy = line.strip().split(', ')
        accuracy_dict[filename] = float(accuracy)

    # Update accuracy dictionary with data from desnet, keeping the higher accuracy
    for line in id_est_data:
        filename, accuracy = line.strip().split(', ')
        accuracy = float(accuracy)
        if filename in accuracy_dict and accuracy_dict[filename] > 0.8:
            accuracy_dict[filename] = max(accuracy_dict[filename], accuracy)
        else:
            accuracy_dict[filename] = accuracy

    # Write the merged accuracies to the output file
    with open(output_file, 'w') as output:
        for filename, accuracy in accuracy_dict.items():
            output.write(f"{filename}, {accuracy}\n")
            
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_paths = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),  # Assuming DenseNet input size is 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom DataLoader
    #step2: check correct dir of images being tested
    image_dir = "eval_simulator/image_folder"
    image_names = os.listdir(image_dir)
    image_dataset = ImageDataset(image_dir, transform=preprocess)
    data_loader = DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
    #load model
    #step3: mode name should be correct no x2
    model = densenet121.DenseNet121(num_classes=42).to(device)
    #step4: correct name of weight
    PATH = './best_model_parameters_densenet121_run3.pth'
    model.load_state_dict(torch.load(PATH))

    #get non smu index
    non_smu_index = 21

    #create text file
    #step5: rename text file correctly
    ranking_result_file = "id_est.txt_201_run3_best2"
    # ranking_result_file = "dummy_results_model.txt"
    ranking_result = open(ranking_result_file, "w")
    # Iterate through the DataLoader and make predictions
    model.eval()
    i = 0
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)

        # Post-process predictions as needed
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        for p in probabilities:
            likelihood_smu = 1- p[non_smu_index]
            ranking_result.write(f"{image_names[i]}, {likelihood_smu}\n")
            i += 1
    ranking_result.close()
    print("initial ranking completed")

    # # File Dependancy paths
    # logos_file = open("logos.txt", "r")
    # ranking_file = open(ranking_result_file, "r")
    # output_file = open("id_est.txt", "w")
    # Iterate over both files simultaneously
    # for logos_line, id_est_line in zip(logos_file, ranking_file):
    #         # Strip lines to remove any leading/trailing whitespace
    #         logos_line = logos_line.strip()
    #         id_est_line = id_est_line.strip()
            
    #         # Extracts chance of logo detection
    #         logos_value = float(logos_line.split()[-1])
    #         # logos_file line overries id_est_line if logos_value exceeds threshold
    #         if logos_value > 0.8:
    #             id_est_value = float(id_est_line.split()[-1])
    #             if logos_value > id_est_value:
    #                 output_file.write(logos_line + "\n")
    #         else:
    #             output_file.write(id_est_line + "\n")

    # logos_file.close()
    # ranking_file.close()
    # output_file.close()
    
    # File paths
    logos_file = 'logos.txt'
    id_est_file = 'dummy_results_model.txt'
    output_file = 'output_editfile.txt'



    print("final ranking completed")