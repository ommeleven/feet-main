from footfracture.config import DATA_DIR
import pandas as pd
from collections import Counter
import shutil
import os

# Input CSV file
input_file = os.path.join(DATA_DIR, "cleaned_annotations.csv")  # Replace with your input file name

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Initialize a Counter to count labels
label_counter = Counter()

# Source directory of images
image_dir = os.path.join(DATA_DIR, "ANKLE SAGIR")

# Target directory for storing images based on label
target_dir = os.path.join(DATA_DIR, "new_data/all_images")

# Process each row in the DataFrame
for _, row in df.iterrows():
    # Get the image name
    image_name = row["Image"]
    
    # Check if the image file exists
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found!")
        continue
    
    # Split annotations if there are multiple (they are separated by a semicolon)
    annotations = row["Annotations"].split("; ")
    
    # Initialize a set to track labels in the image
    image_labels = set()
    
    for annotation in annotations:
        # Extract label (the first part before the parenthesis)
        label = annotation.split(" ")[0]
        image_labels.add(label)
    
    # If both "fracture" and "edema" are present, count it as "fracture"
    if "fracture" in image_labels:
        label_counter["fracture"] += 1
        target_path = os.path.join(target_dir, "fracture")
    elif "edema" in image_labels:
        label_counter["edema"] += 1
        target_path = os.path.join(target_dir, "edema")
    else:
        label_counter["healthy"] += 1
        target_path = os.path.join(target_dir, "healthy")
        
    # Make sure the target directory exists, otherwise create it
    os.makedirs(target_path, exist_ok=True)
    
    # Copy the image to the corresponding directory
    shutil.copy(image_path, os.path.join(target_path, image_name))

# Print the counts of each label
print("Label Counts:")
for label, count in label_counter.items():
    print(f"{label}: {count}")
