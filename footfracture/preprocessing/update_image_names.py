import os
import shutil

import pandas as pd
from footfracture.config import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'))
  
# use thsi fucntion to populate healthy image dataset folder
def get_png_full_path(file):
  """
  Finds the full path of a PNG file with the given filename.

  Args:
      filename (str): Name of the PNG file.

  Returns:
      str: Full path of the file if found, None otherwise.
  """
  
  current_dir = DATA_DIR + "/images"
  healthy_folder = os.path.join(DATA_DIR, '864', 'train','healthy')
  count = 0
  for root, dirs, files in os.walk(current_dir):
      if files:
          for filename in files:
              if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                  if filename not in df["image"].values:
                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(healthy_folder, filename)
                    #print(src_path)
                    
                    
                    #print(dest_path)
                    shutil.move(src_path, dest_path)
                    print(f"Moved {filename} to healthy folder.")
                    return 1 
                  
                  #image_path = os.path.join(root, filename)
                  #return image_path
  return -1
  #return None
  '''
def move_images_to_healthy_folder(DATA_DIR, df):
    # DATA_DIR/864/validate
    healthy_folder = os.path.join(DATA_DIR, '864', 'validate',"healthy")
    os.makedirs(healthy_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(DATA_DIR):
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
print(get_png_full_path("P001 SAGT1_020.jpg"))
len = 0
for index, row in df.iterrows():
   #print(row['image'])
   len += get_png_full_path(row['image'])
print(len)
#move_images_to_healthy_folder(DATA_DIR, df)                  #
#print(df.head())
#df.to_csv(os.path.join(DATA_DIR, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv'), index=False)

#filename = "P001 SAGT1_008.jpg"
#full_path = get_png_full_path(filename)

  #print("Full path:", full_path)
#else:
  #print("PNG file not found.")
