# -*- coding: utf-8 -*-
from footfracture.config import DATA_DIR
"""
Created on Wed Jul 10 9:37:36 2024

@author: Anshuman
"""

import os
import cv2

def resize_and_save_image(image_path, output_folder, new_size=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}") 
        return
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    relative_path = os.path.relpath(image_path, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized_image)
    print(f"Resized and saved image: {output_path}")

def process_images(input_folder, output_folder, new_size=(512, 512)):
    for root, dirs, files in os.walk(input_folder):
        if root.endswith('SAGT1'):
            print(f"Processing folder: {root}")
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                for subroot, _, subfiles in os.walk(subdir_path):
                    for file in subfiles:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                            image_path = os.path.join(subroot, file)
                            #print(f"Resizing image: {image_path}")
                            image = cv2.imread(image_path)
                            relative_path = os.path.relpath(image_path, input_folder)
                            output_path = os.path.join(output_folder, relative_path)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            cv2.imwrite(output_path, image)
                            #resize_and_save_image(image_path, output_folder, new_size)

input_folder = DATA_DIR
output_folder = os.path.join(DATA_DIR, "SAGT1s")
process_images(input_folder, output_folder)
