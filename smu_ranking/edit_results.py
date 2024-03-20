# logos_file = open("logos.txt", "r")
# id_est_file = open("dummy_results_model.txt", "r")
# output_file = open("output_editfile.txt", "w")

# logos_lines = []
# for l in logos_file.readlines():
#     logos_lines.append(l.strip())
# id_est_lines = id_est_file.readlines()

# for line in id_est_lines:
#     split_line = line.split(", ")
#     filename = split_line[0]
#     accuracy = split_line[1]
#     if filename in logos_lines:
#         print(filename)
#         output_file.write(filename + " 1.0\n")
#     else:
#         output_file.write(line)

# logos_file.close()
# id_est_file.close()
# output_file.close()

def merge_accuracies(logos_file, id_est_file, output_file):
    # Read data from the logo detection
    with open(logos_file, 'r') as logos:
        logos_data = logos.readlines()

    # Read data from desnet
    with open(id_est_file, 'r') as id_est:
        id_est_data = id_est.readlines()

    # Create a dictionary to store accuracies for each image
    accuracy_dict = {}

    # Populate accuracy dictionary from logos.txt
    for line in logos_data:
        filename, accuracy = line.strip().split(', ')
        accuracy_dict[filename] = float(accuracy)

    # Update accuracy dictionary with data from desnet, keeping the higher accuracy
    for line in id_est_data:
        filename, accuracy = line.strip().split(', ')
        accuracy = float(accuracy)
        if filename in accuracy_dict and accuracy_dict[filename] > 0.8:
            accuracy_dict[filename] = max(accuracy_dict[filename], accuracy)
            print(accuracy_dict[filename], accuracy)
        else:
            accuracy_dict[filename] = accuracy

    # Write the merged accuracies to the output file
    with open(output_file, 'w') as output:
        for filename, accuracy in accuracy_dict.items():
            output.write(f"{filename}, {accuracy}\n")


# File paths
logos_file = 'logos.txt'
id_est_file = 'dummy_results_model.txt'
output_file = 'output_editfile.txt'

# Merge accuracies and write to output file
merge_accuracies(logos_file, id_est_file, output_file)