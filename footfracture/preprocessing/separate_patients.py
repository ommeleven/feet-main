from footfracture.config import DATA_DIR
import os
import random

def get_files_in_directory(directory):
    # List all files in the directory
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Example usage
directory = os.path.join(DATA_DIR, "all-sagir-images/healthy(non black)")  # Specify your folder path
file_names = get_files_in_directory(directory)
new_list = dict()
for name in file_names:
    new_list[name[:4]] = []

for name in file_names:
    new_list[name[:4]].append(name)


keys = list(new_list.keys())
random.shuffle(keys)
length = len(keys)
#print(length)

seventy_percent_length = int(length * 0.8) 
train_keys = keys[:seventy_percent_length]
train_imgs = []
for key in train_keys:
    for img in new_list[key]:
        train_imgs.append(img)

test_keys = []
for key in keys:
    if key not in train_keys:
        test_keys.append(key)



test_imgs = []
for key in test_keys:
    for img in new_list[key]:
        test_imgs.append(img)
print(len(test_imgs))
"""
flattened_train_list = [item for sublist in train_imgs for item in sublist]
print(len(flattened_train_list))

flattened_test_list = [item for sublist in test_imgs for item in sublist]
print(len(flattened_test_list))
"""

import os
import shutil

def find_and_copy_file(filename, source_dir, target_dir):
    # Check if the file exists in the source directory
    for root, dirs, files in os.walk(source_dir):
        if filename in files:
            # Construct the full source file path
            source_file = os.path.join(root, filename)
            # Construct the full target file path
            target_file = os.path.join(target_dir, filename)
            
            # Copy the file to the target directory
            shutil.copy(source_file, target_file)
            print(f"File {filename} copied to {target_dir}")
            return
    print(f"File {filename} not found in {source_dir}")

# Example usage
source_directory = os.path.join(DATA_DIR, "all-sagir-images/healthy(non black)")
target_directory = os.path.join(DATA_DIR, "seperated_patients/train/healthy")
for file_to_find in train_imgs:
    find_and_copy_file(file_to_find, source_directory, target_directory)

