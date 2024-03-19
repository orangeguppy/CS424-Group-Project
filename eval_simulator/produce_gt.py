import pandas as pd

est_file = 'id_est_tst.txt'
gt_dataframe = pd.read_csv(est_file, header = None, sep = ' ', engine='python')
# split it into names and labels
gt_names = list(gt_dataframe[0])

output_file = open("id_gt_tst.txt", "w")
new_file = open("id_est_tst2.txt", "w")
count = 0
for l in gt_names:
    value = 1
    if l[0].isdigit() or "wrong" in l:
        print(l)
        value = 0
    output_file.write(l + ", " + str(value) + "\n")
    new_file.write(l + ", " + str(gt_dataframe[1][count]) + "\n" )
    count += 1

output_file.close()