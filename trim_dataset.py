import os
import shutil

# PROJECT DATASET FOLDER
DATASET_PATH = "dataset"

# Breeds we want
keep = [
"maltese_dog",
"pomeranian",
"samoyed",
"bernese_mountain_dog",
"afghan_hound",
"beagle"
]

folders = os.listdir(DATASET_PATH)

for folder in folders:

    breed = folder.split("-")[-1]

    if breed not in keep:
        path = os.path.join(DATASET_PATH, folder)
        print("Deleting:", folder)
        shutil.rmtree(path)

print("DONE — Only 6 breeds remain")