import os
import shutil

import pandas as pd
from path import my_path

df = pd.read_csv(os.path.join(my_path, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'))
  
# use thsi fucntion to populate healthy image dataset folder
def get_png_full_path(file):
  """
  Finds the full path of a PNG file with the given filename.

  Args:
      filename (str): Name of the PNG file.

  Returns:
      str: Full path of the file if found, None otherwise.
  """
  
  current_dir = my_path
  count = 0
  for root, dirs, files in os.walk(current_dir):
      if files:
          for filename in files:
              if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename == file:
                  count +=1 
                  
                  image_path = os.path.join(root, filename)
                  #return image_path
  return count
  #return None
  '''
def move_images_to_healthy_folder(my_path, df):
    #/Users/HP/src/feet_fracture_data/864/validate
    healthy_folder = os.path.join(my_path, '864', 'validate',"healthy")
    os.makedirs(healthy_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(my_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                if filename not in df["image"].values:
                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(healthy_folder, filename)
                    #print(src_path)
                    
                    
                    #print(dest_path)
                    shutil.move(src_path, dest_path)
                    print(f"Moved {filename} to healthy folder.")
'''
print(get_png_full_path("P001 SAGT1_002.jpg"))
#move_images_to_healthy_folder(my_path, df)                  #
#print(df.head())
#df.to_csv(os.path.join(my_path, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'), index=False)

#filename = "P001 SAGT1_008.jpg"
#full_path = get_png_full_path(filename)

  #print("Full path:", full_path)
#else:
  #print("PNG file not found.")
