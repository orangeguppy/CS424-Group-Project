import torch
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import os
import cv2

def create_test_res(img_dir, device, model, classes):
    # Path to the folder containing images
    folder_path = img_dir

    # List to store images
    images = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Read the image
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is not None:
            # Append the image to the list
            images.append(img)

    non_smu_index = classes.index("non smu")
    print('non smu index: {non_smu_index}')

    #create text file
    f = open("id_est.txt", "w")

    #do the predictions
    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            print(outputs.data)
            #gives highest prob, predicted class
            _, predict = torch.max(outputs.data, 1)
            #list of probabilities for each 
            probabilities = F.softmax(outputs.data, 1)
            for p in probabilities:
                likelihood_smu = 1- p[non_smu_index]
                #return this and add to text file
            res = zip(_, predict)
            del images, labels, outputs
    
    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))
    f.close()
    