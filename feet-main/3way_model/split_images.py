import os
import numpy as np
from sklearn.model_selection import train_test_split
from path import my_path
import shutil

# Assuming you have already loaded your image dataset and labels
X = my_path + '/864/train'# Your image data
y =  my_path + '/864/test' # Your labels (e.g. 'edema', 'fracture', 'healthy' for each class)
all_images = my_path + '/864/all images/healthy'
data  = os.listdir(all_images)
# Split the dataset into training and testing sets (80% train, 20% test)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train, valid = train_test_split(data, test_size=0.2, random_state=1)
#print(valid)
for image in train:
    source = all_images + '/' + image
    des = X + f'/healthy' + '/' + image
    shutil.copy(source, des)
    print('done')
print(f"Training data: {len(train)} images")
# Now you have your training and testing data ready to be used for training and evaluation
