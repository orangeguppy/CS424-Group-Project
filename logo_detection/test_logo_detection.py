import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def test_model(dir, filename):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_model_instance_segmentation(2)

    # move model to the right device
    model.to(device)

    model.load_state_dict(torch.load('logo_detection/train_model_weights.pth'))

    # image = read_image("dataset/smu_logo/images/20240225_142819.png")
    image = read_image(dir, ImageReadMode.RGB)

    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # Get the index of the prediction with the best score
    best_index = torch.argmax(pred["scores"])
    best_label = pred["labels"][best_index]
    best_score = pred["scores"][best_index]
    best_box = pred["boxes"][best_index]

    # Draw the bounding box with the best score
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    output_image = draw_bounding_boxes(image, best_box.unsqueeze(0), [f"smu_logo: {best_score:.3f}"], colors="red")

    # Draw the segmentation mask if present
    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imsave(f"dataset/smu_logo/test_predictions/{filename}", output_image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(output_image.permute(1, 2, 0))

    print("That's it!")

for filename in os.listdir("dataset/smu_logo/test_images"):
    # Check if the path is a file (not a directory)
    if os.path.isfile(os.path.join("dataset/smu_logo/test_images", filename)):
        # Print the filename
        print(filename)
        test_model(f"dataset/smu_logo/test_images/{filename}", filename)