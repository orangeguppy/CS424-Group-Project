def list_folders(directory):
    # List all items (files and folders) in the given directory
    items = os.listdir(directory)
    
    # Iterate through each item
    for item in items:
        # Check if the item is a folder
        if os.path.isdir(os.path.join(directory, item)):
            print(item)