import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(root_dir, train_dir, test_dir, test_size=0.2, seed=42):
    random.seed(seed)
    
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            images = os.listdir(label_dir)
            train_images, test_images = train_test_split(images, test_size=test_size, random_state=seed, stratify=[label]*len(images))
                
            # Move images to train directory
            for image_name in train_images:
                src = os.path.join(label_dir, image_name)
                dst = os.path.join(train_dir, label, image_name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
            
            # Move images to test directory
            for image_name in test_images:
                src = os.path.join(label_dir, image_name)
                dst = os.path.join(test_dir, label, image_name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
                
def main():
    root_dir = '/Users/HP/src/feet_fracture_data/864/all_images'
    train_dir = '/Users/HP/src/feet_fracture_data/864/train_st'
    test_dir = '/Users/HP/src/feet_fracture_data/864/test_st'
    
    split_dataset(root_dir, train_dir, test_dir)

if __name__ == '__main__':
    main()
