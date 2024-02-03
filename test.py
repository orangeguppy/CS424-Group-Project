from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create a PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Follow the authentication process
drive = GoogleDrive(gauth)

# Define the file path of the nested folder
file_path = "/CS424 proj/images/smu logo"

# Split the file path to get folder names
folder_names = file_path.split("/")

# Function to get folder ID by name
def get_folder_id_by_name(folder_name, parent_id=None):
    query = f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    folder_list = drive.ListFile({'q': query}).GetList()
    if folder_list:
        return folder_list[0]['id']
    else:
        return None

# Get the ID of the root folder
root_folder_id = 'root'

# Iterate through each folder name to navigate to the desired folder
current_parent_id = root_folder_id
for folder_name in folder_names:
    folder_id = get_folder_id_by_name(folder_name, current_parent_id)
    if folder_id:
        current_parent_id = folder_id
    else:
        print(f"Folder '{folder_name}' not found.")
        break

# Retrieve all folders inside the nested folder
if current_parent_id:
    folder_list = drive.ListFile({'q': f"'{current_parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed=false"}).GetList()
    print("Folders inside the nested folder:")
    for folder in folder_list:
        print("Folder Name:", folder['title'])
        print("Folder ID:", folder['id'])
else:
    print("Unable to retrieve folders inside the nested folder.")
