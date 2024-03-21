import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os
import numpy as np
import real_logo_utils

"""
Create and load model from trained weights
"""
def create_model(model_weights_path, device):
    model = real_logo_utils.get_model_instance_segmentation(2).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    return model

"""
Draw bounding boxes on the image with a minimum confidence threshold and remove overlapping boxes.
"""
def draw_bounding_boxes_with_threshold(image, boxes, labels, scores, threshold, colors="red"):
    filtered_boxes, filtered_labels, filtered_scores = real_logo_utils.get_bounding_boxes(image, boxes, labels, scores, threshold)
    print(filtered_boxes)
    if filtered_boxes is None:
        return image

    pred_labels = [f"smu_logo: {score:.3f}" for label, score in zip(filtered_labels, filtered_scores)]
    output_image = draw_bounding_boxes(image, filtered_boxes, pred_labels, colors="red")

    return output_image

"""
Generate bounding boxes and draw over images.
"""
def test_model(img_path, filename, epoch, threshold):
    image = read_image(img_path, ImageReadMode.RGB)
    eval_transform = real_logo_utils.get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        print(type(x))
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # Draw the bounding box with the best score
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [f"smu_logo: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()

    # output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
    output_image = draw_bounding_boxes_with_threshold(image, pred_boxes, pred_labels, pred["scores"], threshold=threshold)

    plt.figure(figsize=(12, 12))
    plt.imsave(f"eval_simulator/test_predictions/{filename}", output_image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(output_image.permute(1, 2, 0))

"""
Return all bounding boces for each image
"""
def output_box_coordinates(device, model, image, filename, epoch, threshold):
    eval_transform = real_logo_utils.get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # Draw the bounding box with the best score
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [f"smu_logo: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    filtered_boxes, filtered_labels, filtered_scores = real_logo_utils.get_bounding_boxes(image, pred_boxes, pred_labels, pred["scores"], threshold)
    
    return filtered_boxes, filtered_scores

def draw_boxes_over_images(root_dir, model_weights_path, threshold):
    for filename in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, filename)):
            img_path = root_dir + "/" + filename
            test_model(img_path, filename, model_weights_path, threshold)

def output_box_coords_txt(device, model, image, filename, threshold):
    boxes, scores = output_box_coordinates(device, model, image, filename, 9, threshold)

    if boxes is not None:
        with open("object_est", "a") as file:
            for box in boxes:
                x, y, w, h = box.tolist()  # Convert tensor to list
                # Convert x, y, w, h to the specified format
                # Calculate x, y as the top-left corner coordinates
                x = int(x)
                y = int(y)
                w = int(w - x)  # Convert width to the actual width by subtracting x
                h = int(h - y)  # Convert height to the actual height by subtracting y
                file.write(f"{filename}, {x}, {y}, {w}, {h}\n")
                print(filename, box)

    if len(scores) > 0:
        with open("object_probs", "a") as file:
            score = max(scores)
            file.write(f"{filename}, {score}\n")

# ----------------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------------
threshold = 0.7
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# root_dir = "dataset/smu_logo/new_test"
root_dir = "eval_simulator\image_folder"
model_weights_path = "logo_detection/train_model_weights_epoch_4 (2).pth"
model = create_model(model_weights_path, device)

# ----------------------------------------------------------------------------
# This code snippet draws bounding boxes over all images in a root directory. Change the directory name to specify the 
# root directory with the images
# ----------------------------------------------------------------------------
draw_boxes_over_images(root_dir, model_weights_path, threshold)

# ----------------------------------------------------------------------------
# This code snippet prints out bounding box coordinates like in Prof's code to txt
# ----------------------------------------------------------------------------
for filename in os.listdir(root_dir):
    img_path = root_dir + "/" + filename
    image = read_image(img_path, ImageReadMode.RGB)
    print(type(image))
    output_box_coords_txt(device, model, image, filename, threshold)