import os 
import shutil
from path import my_path

def duplicate_images(root):
    
    images = os.listdir(root)
    for i,image in enumerate(images):
        source = root + '/' + image
        des = root + '/' + f'{i}{i}_{image}'
        #print(des)
        shutil.copy(source, des)

    print('done')

#root = my_path + '/864/train_oversampling/fracture'
root = '/Users/HP/src/feet_fracture_data/864/train/healthy'
duplicate_images(root)