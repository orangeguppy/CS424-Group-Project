import json
import requests
from PIL import Image
from io import BytesIO

import urllib.request

headers = {
    'Authorization': "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHM2Y3Vqa2MwbjFqMDd6aTJ1a3NoeTdlIiwib3JnYW5pemF0aW9uSWQiOiJjbHM2Y3VqazMwbjFpMDd6aTh4bGFhNnl5IiwiYXBpS2V5SWQiOiJjbHM2ZjAzZG0wb2F5MDd2NDZudmRnN2w1Iiwic2VjcmV0IjoiZjk3MDU2NTIwMzM1Mzc1MDdhYzljOWFjMDgyMDRhMzgiLCJpYXQiOjE3MDY5ODUzNDcsImV4cCI6MjMzODEzNzM0N30.kF5GNci2lnhUQX8Sg320LY8R4VLOzWZmGHqmKhBfvzM"
}

# Open the JSON file
with open('logo_masks.json', 'r') as file:
    # Load the JSON data into a Python dictionary
    data = json.load(file)

for img in data["exports"]:
    img_id = img["data_row"]["external_id"]
    mask_url = img["projects"]['clt17dv9z0jaf07yjezdea0p4']["labels"][0]["annotations"]["objects"][0]["mask"]["url"]
    
    # Make the API request
    req = urllib.request.Request(mask_url, headers=headers)

    # Optionally, print the image of the mask
    image = Image.open(urllib.request.urlopen(req))
    # Save the image as a PNG file
    image.save(f"dataset/smu_logo/masks/{img_id}")

# install latest labelbox version (3.0 or above)
# !pip3 install labelbox[data]

# import labelbox
# import json
# # Enter your Labelbox API key here
# LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHM2Y3Vqa2MwbjFqMDd6aTJ1a3NoeTdlIiwib3JnYW5pemF0aW9uSWQiOiJjbHM2Y3VqazMwbjFpMDd6aTh4bGFhNnl5IiwiYXBpS2V5SWQiOiJjbHM2ZjAzZG0wb2F5MDd2NDZudmRnN2w1Iiwic2VjcmV0IjoiZjk3MDU2NTIwMzM1Mzc1MDdhYzljOWFjMDgyMDRhMzgiLCJpYXQiOjE3MDY5ODUzNDcsImV4cCI6MjMzODEzNzM0N30.kF5GNci2lnhUQX8Sg320LY8R4VLOzWZmGHqmKhBfvzM"
# # Create Labelbox client
# lb = labelbox.Client(api_key=LB_API_KEY)
# # Get project by ID
# project = lb.get_project('clt17dv9z0jaf07yjezdea0p4')
# # Export image and text data as an annotation generator:
# # labels = project.label_generator()
# # # Export video annotations as an annotation generator:
# # labels = project.video_label_generator()
# # Export labels created in the selected date range as a json file:
# labels = project.export_labels(download = True, start="2023-02-21", end="2024-02-27")

# with open("logo_masks.json", "w") as outfile:
#     json.dump(labels, outfile)