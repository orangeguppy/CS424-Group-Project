import utils
# images folder id: 1DJOmDUKHTKQBt3bd9vRnsEmigk8a6Ogs
utils.download_dataset("dataset/smu_images/sol_interior","1A5o_MgR02A4dMK0lNg8XLjx30859iDm6", False)
utils.download_dataset("dataset/smu_images/sol_exterior","1VogeIAtM5m2vehZiBYARsVD6dimKe8HV", False)

utils.generate_dataset("dataset/smu_images")
