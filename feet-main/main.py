
import zipfile

# zip_file = '/home/odange/repo/feet-main/864.zip'
zip_file = '/home/odange/repo/feet-main'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(zip_file)
    print("The file was extracted successfully")
    