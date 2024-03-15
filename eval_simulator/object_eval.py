import cv2
import numpy as np
import pandas as pd

"""
Object detection boxes come in one of two forms: coord_boxes or widthheight_boxes.
- Coord_boxes have the form (x1, y1, x2, y2) where (x1, y1) is the coordinate of
  the box's top left corner and (x2, y2) is the coordinate of the box's bottom 
  right hand corner.
- widthheight_boxes have the form (x1, y1, w, h) where (x1, y1) is the coordinate of
  the box's top left corner and (w, h) is the box's resepctive width and height.

Widthheight format is used for groundtruth labels and drawbox which dispalys boxes.
Coord format is used in bb_intersection_over_union() to compute intersection over union.

The functions coordboxes_to_widthheightboxes() and widthheightboxes_to_coordboxes()
convert list of boxes from one format to another. 

"""


"""
Functions to convert one box format into another
"""

def coordboxes_to_widthheightboxes(boxes):
    # boxes is a list of boxes in coord format

    new_boxes = []
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        width = x2 - x1
        height = y2 - y1

        new_boxes.append([x1, y1, width, height])
    return new_boxes


def widthheightboxes_to_coordboxes(boxes):
    # boxes is a list of boxes in widthheight format

    new_boxes = []
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        w = box[2]
        h = box[3]

        x2 = x1 + w
        y2 = y1 + h

        new_boxes.append([x1, y1, x2, y2])
    return new_boxes

"""
Functions to display object detection boxes.
"""


def drawbox(img, boxes=[], display=True, color = (0,0,255), name=''):
    # boxes in a list of boxes in width height format
    # color indicates the desired box color in openCV format
    
    if isinstance(img, str):
        img = cv2.imread(img)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)


    if display is True:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

"""
Evaluation object detection functions.
"""


def bb_intersection_over_union(boxA, boxB):
    # boxA and boxB are two boxes in coordinate format

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def find_match(data1, data2, iou_thres):
    # Compares the pandas dataframes data1 and data2.
    # For the ith bounding box data1, we seek a corresponding box in data2. 
    # If corrspndance is discovered, found[i] = 1; else found[i] = 0.

    # number of bounding boxes in data1
    num = len(data1)
    # initialize found
    found = np.zeros(num, dtype=int)
    # initialize used to prevent multiple assignments to one box
    used = np.zeros(len(data2), dtype=int)
    # first column of data2 contains image names.
    matching_names = data2[0]

    # for each box in data1
    for i in range(num):
        # get box statistics
        line = list(data1.iloc[i])
        # image where box resides
        name = line[0]
        # convert box from widthheight to coord format
        box = widthheightboxes_to_coordboxes([line[1:]])
        # boxes in data2 with the same image name
        mask = np.where(list(matching_names.isin([name])))[0]
        if mask.size==0:
            # no potential correspondances
            continue
        else:
            # for each potential correspondance
            for j in mask:
                # get correspondance statistics
                line_ = list(data2.iloc[j])
                # convert box from widthheight to coord format
                box_ = widthheightboxes_to_coordboxes([line_[1:]])
                # compute intersection over union
                iou = bb_intersection_over_union(box[0], box_[0])
                # accept if iou is sufficiently high 
                if iou > iou_thres and used[j]==0:
                    found[i] = 1 
                    used[j] = 1
    return found


def evaluate(gt_file, est_file, iou_thres=0.5, display=False):
    # Determine the recall, precision and f1 score by comparing ground_truth 
    # bounding boxes with the estimated bounding boxes.
    # see gt.txt and est.txt for examples of bounding box file formats. 

    gt = pd.read_csv(gt_file, header = None, sep = ',', engine='python')
    est = pd.read_csv(est_file, header = None,  sep = ',', engine='python')
    
    # find boxes in gt which appear in est
    found12 = find_match(gt, est, iou_thres)
    recall =  np.mean(found12)
    # find boxes in est which appear in gt
    found21 = find_match(est, gt, iou_thres)
    precision = np.mean(found21)
    # f1 score weights recall and precision equally
    f1 = 2 * recall * precision / (precision + recall)
    
    if display:
        print('recall:', recall)   
        print('precision:', precision)
        print('f1:', f1)

    return recall, precision, f1

            
