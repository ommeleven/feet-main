import os
import shutil
import pandas as pd

# Path to the directory containing all images
source_dir = "path/to/source_directory"

# Path to the directory where images will be copied based on labels
fracture_dir = "/Users/HP/src/feet_fracture_data/864/validate/fracture"
edema_dir = "/Users/HP/src/feet_fracture_data/864/validate/edema"

# Load your DataFrame (replace 'data.csv' with your actual CSV file)
df = pd.read_csv("/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv")
sorted_df=  df.sort_values(by='value_rectanglelabels')
#print(sorted_df['value_rectanglelabels'].head())
# Iterate through DataFrame
for index, row in sorted_df.iterrows():
    #print(f"Processing row {index + 1}/{len(df)}")
    image_name = row['image']  # Assuming 'image_name' is the column containing image filenames
    #print(image_name)
    label = str(row['value_rectanglelabels']) # Assuming 'value_rectanglelabels' contains labels ('fracture' or 'edema')
    #print(label)
    # Source path of the image
    source_path = row['image_path']
    #print(source_path)
    # Destination directory based on label
    label = label[2:-2]
    
    if label == 'Fracture':
        print('fracture')
        destination_dir = fracture_dir
    elif label == 'Edema':
        print('edema')
        destination_dir = edema_dir
    else:
        continue  # Skip if label is not 'fracture' or 'edema'
    
    '''
    print(label, label == 'Edema')
    destination_dir = 'Users/HP/src/feet_fracture_data/864/validate/'+ label.lower()
    #print(destination_dir)
    '''
    # Copy image to the destination directory
    shutil.copy(source_path, destination_dir)

    print(f"Copied {image_name} to {destination_dir}")
