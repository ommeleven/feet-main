import os
import shutil

import pandas as pd
from path import my_path

df = pd.read_csv(os.path.join(my_path, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'))
  

def get_png_full_path(file):
  """
  Finds the full path of a PNG file with the given filename.

  Args:
      filename (str): Name of the PNG file.

  Returns:
      str: Full path of the file if found, None otherwise.
  """
  current_dir = my_path
  for root, dirs, files in os.walk(current_dir):
      if files:
          for filename in files:
              if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename == file:
                  image_path = os.path.join(root, filename)
                  return image_path
  return None

def move_to_healthy_folder(image_path, filename):
    """
    Moves the image to the 'healthy_images' folder.

    Args:
        image_path (str): Full path of the image.
        filename (str): Name of the image file.
    """
    # /Users/HP/src/feet_fracture_data/864/validate
    healthy_images_folder = os.path.join(my_path, '864', 'validate','healthy')
    if not os.path.exists(healthy_images_folder):
        os.makedirs(healthy_images_folder)
    shutil.copy(image_path, os.path.join(healthy_images_folder, filename))

# Test the function
for index, row in df.iterrows():
    image_filename = row['image']
    image_path = get_png_full_path(image_filename)
    if image_path:
        df.at[index, 'image_path'] = image_path
    else:
        move_to_healthy_folder(image_filename, os.path.basename(image_filename))

print(df.head())
df.to_csv(os.path.join(my_path, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'), index=False)

#filename = "P001 SAGT1_008.jpg"
#full_path = get_png_full_path(filename)

  #print("Full path:", full_path)
#else:
  #print("PNG file not found.")
