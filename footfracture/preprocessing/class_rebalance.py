import os 
import shutil
from footfracture.config import DATA_DIR

def duplicate_images(root):
    
    images = os.listdir(root)
    for i,image in enumerate(images):
        source = root + '/' + image
        des = root + '/' + f'{i}{i}_{image[:]}'
        #print(des)
        shutil.copy(source, des)

    print('done')

#root = DATA_DIR + '/864/train_oversampling/fracture'
root = os.path.join(DATA_DIR, "864/train/healthy")
duplicate_images(root)