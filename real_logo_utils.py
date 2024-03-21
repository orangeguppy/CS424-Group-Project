import torch
import numpy as np
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.io import read_image, ImageReadMode

def non_max_suppression(boxes, scores, threshold):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes based on their scores
    indices = np.argsort(scores)
    picked_indices = []

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        picked_indices.append(i)
        suppress = [last]

        for pos in range(last):
            j = indices[pos]
            overlap = calculate_iou(boxes[i], boxes[j])
            if overlap > threshold:
                suppress.append(pos)

        indices = np.delete(indices, suppress)

    return picked_indices

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = box1_area + box2_area - intersection

    return intersection / union

def get_bounding_boxes(image, boxes, labels, scores, threshold):
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    # Filter boxes based on confidence threshold
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    # Apply non-maximum suppression
    indices = non_max_suppression(filtered_boxes, filtered_scores, 0.4)

    # Select elements
    filtered_boxes = [filtered_boxes[i] for i in indices]
    filtered_labels = [filtered_labels[i] for i in indices]
    filtered_scores = [filtered_scores[i] for i in indices]

    # Convert each tuple element to a tensor
    box_tensors = [torch.tensor(t) for t in filtered_boxes]

    while len(filtered_boxes) == 0 and threshold >= 0.5:
        threshold -= 0.1
        # Filter boxes based on confidence threshold
        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)

        # Apply non-maximum suppression
        indices = non_max_suppression(filtered_boxes, filtered_scores, 0.4)

        # Select elements
        filtered_boxes = [filtered_boxes[i] for i in indices]
        filtered_labels = [filtered_labels[i] for i in indices]
        filtered_scores = [filtered_scores[i] for i in indices]

        # Convert each tuple element to a tensor
        box_tensors = [torch.tensor(t) for t in filtered_boxes]

    if len(filtered_boxes) == 0:
        return None, filtered_labels, filtered_scores

    # Stack tensors along a new dimension (if necessary)
    filtered_boxes = torch.stack(box_tensors)
    return filtered_boxes, filtered_labels, filtered_scores

def get_model_scores(dir):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_model_instance_segmentation(2)

    # move model to the right device
    model.to(device)

    model.load_state_dict(torch.load(f'logo_detection/train_model_weights_epoch.pth', map_location=torch.device('cpu')))

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

    scores = pred["scores"]
    return max(scores)

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

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