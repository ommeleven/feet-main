import os
import shutil
import pandas as pd
from path import remote_path

df = pd.read_csv('/home/odange/repo/feet_fracture_data/feet_fracture_data/2-way/train_epoch_49 Labels.csv')


# use this fucntion to populate healthy image dataset folder
def get_unhealthy_imgs(file):
  """
  Finds the full path of a PNG file with the given filename.

  Args:
      filename (str): Name of the PNG file.

  Returns:
      str: Full path of the file if found, None otherwise.
  """
  
  current_dir = remote_path + "/all_images"
  #healthy_folder = os.path.join(my_path, '864', 'train','healthy')
  count = 0

  for root, dirs, files in os.walk(current_dir):
    if files == ['.DS_Store']:
        continue
    #print(root)
    for filename in files:
        if filename.lower().endswith(('.jpg')):
            if file[0] in files:
                print(file[1])
                count += 1
                src_path = os.path.join(root, filename)
                dest_path = os.path.join('/home/odange/repo/feet_fracture_data/feet_fracture_data/2-way/unhealthy_imgs/true_labels/', filename)
                #print(src_path)
                #print(dest_path)
    return count


  for root, dirs, files in os.walk(current_dir):
    if files == ['.DS_Store']:
        continue
    if files:
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                true_label = file[1]
                if filename in df["Image"].values:
                # Check if true label is 1
                    if true_label == 1:
                        src_path = os.path.join(root, filename)
                        dest_path = os.path.join('/home/odange/repo/feet_fracture_data/feet_fracture_data/2-way/unhealthy_imgs/true_labels/', filename)
                        print(src_path)
                        print(dest_path)
                        #shutil.move(src_path, dest_path)
                        #print(f"Moved {filename} to healthy folder.")
                        return 1 
                        
                    #image_path = os.path.join(root, filename)
                    #return image_path
  return -1
  

len = 0
df = df.replace(";", ",", regex=True)
df = df.rename(columns={'Image;True Label;Predicted Label': 'Image'})
#print(df.size)
#print(df.columns)
# df['Image'] = df['Image;True Label;Predicted Label'].apply(lambda x: x.split(";")[0])
#print((df.head))
'
for index, row in df.iterrows():
   filename = row[0]
   #print(filename.split(","))
   len += get_unhealthy_imgs(filename.split(","))
print(len)
