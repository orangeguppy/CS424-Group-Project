import torch
import os
#step1: comment out model not being tested
import densenet201
#import densenet121
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
    image_dir = "eval_simulator/image_folder2"
    image_names = os.listdir(image_dir)
    image_dataset = ImageDataset(image_dir, transform=preprocess)
    data_loader = DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
    #load model
    #step3: mode name should be correct no x2
    model = densenet201.DenseNet201(num_classes=42).to(device)
    #step4: correct name of weight
    #PATH = './densenet201_run3.pth'
    PATH = './best_model_parameters_densenet201_run3.pth'
    model.load_state_dict(torch.load(PATH))

    #get non smu index
    non_smu_index = 21

    #create text file
    #step4: rename text file correctly
    f = open("id_est.txt_201_run3_best2", "w")
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
            f.write(f"{image_names[i]}, {likelihood_smu}\n")
            i += 1
    print("done")









