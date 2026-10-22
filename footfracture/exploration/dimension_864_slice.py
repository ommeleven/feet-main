from footfracture.config import DATA_DIR
import os
import pandas as pd
from PIL import Image


def all_image_864(folder):
    all_864 = []
    tag_csv_pd = pd.read_csv(tag_csv)
    distinct_set = set()
    for root, subdirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.jpg'):

                image_path = os.path.join(root, file)
                
                # Open the image using PIL
                image = Image.open(image_path)
                
                # Get image dimensions (width, height, and channels)
                width, height = image.size
                channels = len(image.getbands())
                if width == 864 and height == 864:
                    tag = ''
                    print(file, width, "good")
                    if file in tag_csv_pd['image'].tolist():
                        
                        tag = 'Positive'
                    else:
                        
                        tag = 'Negative'
                    all_864.append([file, width, height, tag])
                    distinct_set.add(file[0:4])

    df = pd.DataFrame(all_864, columns=['image', 'width', 'height', 'tag'])
    df.to_csv(output_csv_path, index=False)
    print(distinct_set)

def main():
    # bonding_box_folder()
    all_image_864(folder_path)
    

if __name__ == '__main__':
    #folder_path = os.path.join(DATA_DIR, 'raw')
    folder_path = DATA_DIR
    #tag_csv = os.path.join(DATA_DIR, 'json_data', 'sag_T1_json_columns', 'annotations', 'results_box_cleaned_calculation.csv')
    tag_csv = DATA_DIR + "/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv"
    #output_csv_path = os.path.join(DATA_DIR, 'image_basic_data', '864_slices.csv')
    output_csv_path = DATA_DIR + '/image_basic_data/864_slices.csv'
    main()
