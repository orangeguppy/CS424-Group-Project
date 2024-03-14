########################################## Description ####################################################################
Place the SMU-like score of each image in id_est.txt. Each row contains the score of an individual image,
in the form [n, s]:
n being the image name
s being the score

Scores are compared to the groundtruth in id_gt.txt. id_checking_function.ipynb illustrates the process

Example id_gt.txt and id_est.txt files are provided. Delibrate errors have been introduced to illustarte errors. 

########################################## Evaluation Day ####################################################################
On the day of evaluation, use the provided test images to generate a id_est.txt file, which will be used for evaluation. 
