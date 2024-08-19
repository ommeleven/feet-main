import os
import shutil
import pandas as pd
from path import remote_path

df = pd.read_csv('/home/odange/repo/feet_fracture_data/feet_fracture_data/accuracy_predictions/results_box_cleaned_calculation.csv')


# use this fucntion to populate healthy image dataset folder
def get_unhealthy_imgs(image): 
  current_dir = '/home/odange/repo/feet_fracture_data/feet_fracture_data/patient_images'
  #healthy_folder = os.path.join(my_path, '864', 'train','healthy')
  des_path = '/home/odange/repo/feet_fracture_data/feet_fracture_data/SAGT1_Patients'
  count = 0
  for dirpath, dirnames, filenames in os.walk(current_dir):
    if filenames == ['.DS_Store']:
        continue
    print(dirpath)
    #for dirpath1, dirnames1, filenames1 in os.walk(current_dir):
    
    if image in filenames:
        print(f"File found: {os.path.join(dirpath, filename)}")
        return 1
    print("File not found")
    return -1


  '''  
  for root, dirs, files in os.walk(current_dir):
    if files == ['.DS_Store']:
        continue
    
    
    if files:
        #print(files)
        for filename in files:
            if filename.lower().endswith(('.jpg')):
                if file[0] in files:
                    #if file[1] == 1:
                    print(file[0])
                    count += 1
                    src_path = os.path.join(current_dir, filename)
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
''' 

len = 0
#df = df.replace(";", ",", regex=True)
#df = df.rename(columns={'Image;True Label;Predicted Label': 'Image'})
#print(df.size)
#print(df.columns)
# df['Image'] = df['Image;True Label;Predicted Label'].apply(lambda x: x.split(";")[0])
#print((df.head))


count = 0
df = df[:-2]
for index, row in df.iterrows():
   filename = row[0]

   #filename = filename.split(",")
   #if "['Edema']" == df['value_rectanglelabels'][index]:
    #count+= 1

   #break
   

   if "['Edema']" == df['value_rectanglelabels'][index]:
    #print("here")
    current_dir = '/home/odange/repo/feet_fracture_data/feet_fracture_data/patient_images'
    #healthy_folder = os.path.join(my_path, '864', 'train','healthy')
    des_path = '/home/odange/repo/feet_fracture_data/feet_fracture_data/all_images_b/unhealthy'
    
    for dirpath, dirnames, filenames in os.walk(current_dir):
        if filenames == ['.DS_Store']:
            continue       
        if 'MRI' in dirpath:
            images = os.listdir(dirpath)
            if filename in images:
                count += 1
                src_path = os.path.join(dirpath, filename)
                
                #print(src_path, des_path + '/' + filename)
                shutil.copy(src_path, des_path)

                #print(filename)
print(count)
            
    #len += get_unhealthy_imgs(filename[0])

print(count)
   




