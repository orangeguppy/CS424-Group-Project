from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

# Query
query = {'q': "sharedWithMe and title = 'CS424 proj' and trashed = false"}

# Auto-iterate through all files that matches this query
file_list = drive.ListFile(query).GetList()

for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))