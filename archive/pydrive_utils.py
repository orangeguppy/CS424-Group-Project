import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

def get_folders_in_folder(parent_folder_id):
    folder_list = drive.ListFile({'q': f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"}).GetList()
    return folder_list

def get_files_in_folder(parent_folder_id):
    file_list = drive.ListFile({'q': f"'{parent_folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"}).GetList()
    return file_list

# Download all files (NOT folders) to a local directory, from a parent_folder_id
def download_files_to_local_directory(local_dir, parent_folder_id):
    file_list = get_files_in_folder(parent_folder_id)
    # Download images to the local directory
    for file in file_list:
        # print('title: %s, id: %s' % (file['title'], file['id']))
        file = drive.CreateFile({'id': file['id']})
        file.GetContentFile(f"{local_dir}/{file['title']}")
    print('downloaded')

# This is a query to verify that the Python script is able to access the CS424 proj root folder
query = {'q': "sharedWithMe and title = 'CS424 proj' and trashed = false"}
file_list = drive.ListFile(query).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))
  parent_id = file1['id']