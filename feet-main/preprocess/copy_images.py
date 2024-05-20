import os
import shutil

def copy_images_with_sagt1(src_folder, dst_folder):
    # Check if the source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    # Create the destination folder if it does not exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # Iterate over files in the source folder
    for filename in os.listdir(src_folder):
        # Check if the file name contains 'SAGT1' (case-sensitive)
        if 'SAGT1' in filename:
            # Construct full file path
            src_file_path = os.path.join(src_folder, filename)
            dst_file_path = os.path.join(dst_folder, filename)
            
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(src_file_path):
                # Copy the file to the destination folder
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied: {src_file_path} -> {dst_file_path}")
            else:
                print(f"Skipping directory: {src_file_path}")

    print("Operation completed.")

# Example usage
source_folder = '/home/odange/repo/feet_fracture_data/feet_fracture_data/all_images/healthy'
destination_folder = '/home/odange/repo/feet_fracture_data/feet_fracture_data/all_images/healthy_sagt1s'
copy_images_with_sagt1(source_folder, destination_folder)
