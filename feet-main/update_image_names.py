import os

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
  # print(df.head())
  for root, dirs, files in os.walk(my_path):
      if files:
          patient = os.path.basename(os.path.dirname(root))  
          dimensions_set = set()
          for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename in df['image'].tolist():
              image_path = os.path.join(root, filename)
              #print("comp: ", filename, file)
              if filename == file:
                
                df.at[df['image'] == file, 'image'] = 1
                print(df.head())
                print("found: ", image_path)
                return image_path.__str__()
  return "PNG file not found."


# Test the function
filename = "P001 SAGT1_008.jpg"
full_path = get_png_full_path(filename)

if full_path:
  print("Full path:", full_path)
else:
  print("PNG file not found.")
