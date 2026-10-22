from footfracture.config import DATA_DIR
import os
from PIL import Image
import numpy as np
import shutil

# Define paths
input_folder = os.path.join(DATA_DIR, "new_data/all_images/healthy")
output_folder = os.path.join(DATA_DIR, "new data w/healthy")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the threshold logic
def check_black_pixel_percentage(image_path, threshold=65):
    # Open the image and convert it to grayscale
    image = Image.open(image_path).convert('L')
    # Convert image to numpy array
    pixels = np.array(image)
    
    # Count black pixels (<= 10)
    black_pixels = np.sum(pixels <= 10)
    total_pixels = pixels.size
    
    # Calculate percentage of black pixels
    black_pixel_percentage = (black_pixels / total_pixels) * 100
    
    # Check if black pixel percentage meets threshold
    return black_pixel_percentage < threshold

# Process images
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        image_path = os.path.join(input_folder, filename)
        
        # Check the black pixel percentage
        if check_black_pixel_percentage(image_path):
            # Move image to output folder if it meets the criteria
            shutil.copy(image_path, os.path.join(output_folder, filename))
            print(f"{filename} passed and was moved to the output folder.")
        else:
            print(f"{filename} ignored due to high black pixel percentage.")
