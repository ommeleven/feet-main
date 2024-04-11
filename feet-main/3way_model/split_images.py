import os
import shutil
import pandas as pd

# Set paths
image_folder = '/Users/HP/src/feet_fracture_data/864/test/edema/edema'
fracture_folder = "/Users/HP/src/feet_fracture_data/864/test/fracture"
healthy_folder = "/Users/HP/src/feet_fracture_data/864/test/validate"

# Load DataFrame
csv_path = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'
df = pd.read_csv(csv_path)

# List all images in the folder
image_files = os.listdir(image_folder)

# Function to move images to respective folders
def move_images(files, source_dir, fracture_dir, healthy_dir):
    for file in files:
        if file in df['image'].values:
            src = os.path.join(source_dir, file)
            dst = os.path.join(fracture_dir, file)
            shutil.move(src, dst)
        else:
            src = os.path.join(source_dir, file)
            dst = os.path.join(healthy_dir, file)
            shutil.move(src, dst)

# Move images to respective folders
move_images(image_files, image_folder, fracture_folder, healthy_folder)

print("Images moved successfully!")
