import torch
import os
import densenet201
import densenet121
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
    image_dir = "dataset/classification_images/basement glass are"
    image_names = os.listdir(image_dir)
    image_dataset = ImageDataset(image_dir, transform=preprocess)
    data_loader = DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
    #load model
    model = densenet121.DenseNet121(num_classes=35).to(device)
    PATH = './densenet121_run2.pth'
    #PATH = './best_model_parameters.pth'
    model.load_state_dict(torch.load(PATH))

    #get non smu index
    non_smu_index = 21

    #create text file
    f = open("id_est.txt_121_run2_basement glass", "w")
    # Iterate through the DataLoader and make predictions
    model.eval()
    i = 0
    for batch in data_loader:
        with torch.no_grad():
            output = model(batch)

        # Post-process predictions as needed
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        for p in probabilities:
            likelihood_smu = 1- p[non_smu_index]
            f.write(f"{image_names[i]} {likelihood_smu}\n")
            i += 1








