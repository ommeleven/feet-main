import os
import matplotlib.pyplot as plt
import pandas as pd

from path import my_path

def plot_bbox(image_path, x_cd, y_cd, width, height, angle, save_path):
    new_image = plt.imread(image_path)

    # Plot the image
    plt.imshow(new_image)
    
    # Recalculate bounding box coordinates based on image dimensions
    plt.gca().add_patch(plt.Rectangle((x_cd, y_cd), width, height, angle=angle, edgecolor='r', linewidth=1, fill=False))

    # Show the plot
    #plt.show()
    #save_path = os.path.join(my_path, 'bbox_renderings_rotatations', 'lw1_all_rotations', os.path.basename(image_path))
    #save_path = '/Users/HP/src/feet_fracture_data///' + os.path.basename(image_path)
    plt.savefig(save_path)
    plt.clf()

def main():
    csv_path = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        image_path = os.path.join(row['image_path'])  
        image = row['image']
        x_cd = row['x_by_percent']
        y_cd = row['y_by_percent']
        width = row['value_width_percent']
        height = row['value_height_percent']
        angle = row['value_rotation']
        save_path = os.path.join(my_path, 'bbox_renderings_rotatations', 'lw1_all_rotations', f'rbbox_{image}')

       
        # Plot bounding box for each row if there is rotation
        if angle != 0:
            plot_bbox(image_path, x_cd, y_cd, width, height, angle, save_path)

if __name__ == "__main__":
    main()
