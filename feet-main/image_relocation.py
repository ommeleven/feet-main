# relocate images to for all bbox rendering 
import shutil
import os

from path import my_path

def copy_sagt1s():
     
     source_dir = my_path
     for root, dirs, files in os.walk(source_dir):
          print(type(dir))
          for dir in dirs:  
               if dir.endswith('SAGT1'):
                    source_path = os.path.join(root, dir)
                    des_path = os.path.join(my_path, 'bbox_renderings')
                    shutil.copyfile(source_path, des_path)
                    

def main():
     
     copy_sagt1s()

if __name__ == '__main__':
     main()

                    