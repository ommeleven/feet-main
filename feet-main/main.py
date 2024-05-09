import sys
from zipfile import PyZipFile

zip_file = '/home/odange/repo/feet-main/864.zip'
pzf = PyZipFile(zip_file)
pzf.extractall()