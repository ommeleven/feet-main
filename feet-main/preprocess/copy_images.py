import os
import shutil
import random
import pandas as pd
from path import my_path

def relocate_images():
    df = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'
    df = pd.read_csv(df)
    #test_images_nums = int(0.2 * len(df['image_path']))
    
    

relocate_images()