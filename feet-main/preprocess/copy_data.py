import os
import zipfile
import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd

path = '/home/odange/repo/feet-main/feet_fracture_data'

isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    print("The directory was created successfully")

data_path = '/home/odange/repo/feet-main/feet_fracture_data'

isExist = os.path.exists(data_path)
gdd.download_file_from_google_drive(file_id='1UXvNGn5wYT3w5Xg4iach2s8EniM4yGCP', dest_path= './864.zip', unzip=True)
#gdown.download(url='1UXvNGn5wYT3w5Xg4iach2s8EniM4yGCP', output= './feet_data.zip', quiet=False)
download_path = f'/home/odange/repo/feet-main/feet_fracture_data/864.zip'
with zipfile.ZipFile(download_path, 'r') as zip_ref:
    dataset_folder = f'{data_path}/download'
    zip_ref.extractall(dataset_folder)

'''
data_root = '/Users/HP/src/feet_fracture_data/864'
dataset_folder = f'{data_root}/feet_data' 


data_root = '/Users/HP/src/feet_fracture_data/864'
dataset_folder = f'{data_root}/download'

url = 'https://drive.google.com/file/uc?id=1UXvNGn5wYT3w5Xg4iach2s8EniM4yGCP'
download_path = f'{data_root}/864.zip'

if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)

new_path = '/Users/HP/src/feet_fracture_data/864'
gdown.download(url, download_path, quiet=False)

with zipfile.ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_folder)
'''