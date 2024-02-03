from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

def get_folders_in_folder(parent_folder_id):
    folder_list = drive.ListFile({'q': f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"}).GetList()
    return folder_list

def get_files_in_folder(parent_folder_id):
    folder_list = drive.ListFile({'q': f"'{parent_folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"}).GetList()
    return file_list

# Download all files (NOT folders) to a local directory, from a parent_folder_id
def download_files_to_local_directory(local_dir_path, parent_folder_id):
    file_list = get_files_in_folder(parent_folder_id)
    # Download images to the local directory
    for file in file_list:
        file.GetContentFile(os.path.join(local_dir, file['title']))

# This is a query to verify that the Python script is able to access the CS424 proj root folder
query = {'q': "sharedWithMe and title = 'CS424 proj' and trashed = false"}
file_list = drive.ListFile(query).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))
  parent_id = file1['id']

# Declare the Folder ID of the folder containing images of SMU's logo
smu_logo_folder_id = "1AILB_g4xqaMCCo1Ors4X3iTiaxIuuBvB" # smu logo

# Get all folders inside the parent folder
nested_folders = get_files_in_folder(parent_id)

# Print names and IDs of nested folders
for folder in nested_folders:
    print("Folder Name:", folder['title'])
    print("Folder ID:", folder['id'])

# title: CS424 proj, id: 1vOqmT0UW5xDqso3-cy-Hq_14JSSGErPa
# Folder Name: smu logo
# Folder ID: 1AILB_g4xqaMCCo1Ors4X3iTiaxIuuBvB
# Folder Name: sol interior
# Folder ID: 1A5o_MgR02A4dMK0lNg8XLjx30859iDm6
# Folder Name: sol exterior
# Folder ID: 1VogeIAtM5m2vehZiBYARsVD6dimKe8HV
# Folder Name: soa/soe/scis interior
# Folder ID: 1--jOVfADiQxz7Q_xWFRjbc1E2IPFxgAu
# Folder Name: li ka shing exterior
# Folder ID: 1Ll9WTAJA5div9rLlgP5CBuCWmUYeRl77
# Folder Name: li ka shing interior
# Folder ID: 1UR-Oe7rAKBt2iyS3VuSri8nB2xt6T3G8
# Folder Name: sob exterior
# Folder ID: 1IWLv2ycBadU8S76xS1KdKGUIC9gEpYgA
# Folder Name: misc
# Folder ID: 1wPycej5Zt8GIOuuHxJdzTXHrmcONRyAN
# Folder Name: fish tanks
# Folder ID: 1SaTFRiEt809ugetKsOA3ut6Hr_e6Oj3F
# Folder Name: connexion
# Folder ID: 1E4BGO_q3RuLZYz-3xQ-wYybt5NQIJbBZ
# Folder Name: koufu
# Folder ID: 1-2knpre2f4I7dcY7yNKIzudCc8L_veTM
# Folder Name: basement
# Folder ID: 1-2YdIMVaIiqGUJQycxJV-TKTOqtTKcck
# Folder Name: admin building interior
# Folder ID: 1Dy-vvesTY5IL5RWF_9VaFTRLEYW5ikpu
# Folder Name: soss exterior
# Folder ID: 1UEVQcQz_m1E478MKkVE6rZE4v6uEhe53
# Folder Name: soss interior
# Folder ID: 1lP5k5yn5BVnP57wYf6IQpzrThW85iQe5
# Folder Name: law library exterior
# Folder ID: 1WFv5dXRRVi6GYWseTIV_mCE0ztfEfa95
# Folder Name: law library interior
# Folder ID: 1oSeyQFcad_5bnu_6Gw_bAjjBEOnowFHg