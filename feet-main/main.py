import sys
from zipfile import PyZipFile

zip_file = '/home/odange/repo/feet-main/864.zip'
print(f'Extracting {zip_file}...')
pzf = PyZipFile(zip_file)
pzf.extractall()