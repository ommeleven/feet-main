import os
import random
import shutil

def split_subfolders(source_folder, train_folder, validate_folder, split_ratio=0.8):
    # Ensure the split ratio is between 0 and 1
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("Split ratio must be between 0 and 1")

    # Create train and validate folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validate_folder, exist_ok=True)

    # List all subfolders in the source folder
    subfolders = [f.name for f in os.scandir(source_folder) if f.is_dir()]

    # Randomly shuffle the list of subfolders
    random.shuffle(subfolders)

    # Determine how many subfolders go into train and validate
    num_subfolders = len(subfolders)
    num_train = int(num_subfolders * split_ratio)
    num_validate = num_subfolders - num_train

    # Move subfolders to train and validate folders
    for i, subfolder_name in enumerate(subfolders):
        source_subfolder = os.path.join(source_folder, subfolder_name)
        if i < num_train:
            shutil.move(source_subfolder, os.path.join(train_folder, subfolder_name))
        else:
            shutil.move(source_subfolder, os.path.join(validate_folder, subfolder_name))

    print(f"Splitting completed: {num_train} subfolders moved to {train_folder}, {num_validate} subfolders moved to {validate_folder}")

# Example usage:
source_folder = '/home/odange/repo/feet_fracture_data/all images/images'
train_folder = '/home/odange/repo/feet_fracture_data/3_way/train'
validate_folder = '/home/odange/repo/feet_fracture_data/3_way/val'
split_ratio = 0.8  # 80% for training, 20% for validation

split_subfolders(source_folder, train_folder, validate_folder, split_ratio)
