########################################## Description ####################################################################
Detected bounding boxes are to be placed in the file object_est.txt. Each row contains the specifications of an individual box,
in the form [n, x, y, w, d]:
n being the image name
x being the x-coordinate of a box's top left corner
y being the y-coordinate of the box's top left corner
w being the box's width in pixels
d being the box's width in pixels

Detected bounding boxes are compared to the groundtruth in object_gt.txt. Object_checking_function.ipynb illustrates the process

Example object_gt.txt and object_est.txt files are provided. Delibrate errors have been introduced in samples1, samples2 and samples3 of est.txt to illustarte wrong bounding boxes. 

########################################## Evaluation Day ####################################################################
On the day of evaluation, use the provided test images to generate a object_est.txt file, which will be used for evaluation. 
