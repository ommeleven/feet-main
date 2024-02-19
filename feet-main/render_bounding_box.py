import matplotlib.pyplot as plt
import pandas as pd

from path import my_path


def plot_bbox(image_path, x_cd, y_cd, width, height):
    new_image = plt.imread(image_path)

    x_cd = x_cd * 512/100
    y_cd = y_cd * 512/100
    width = width * 512/100
    height = height * 512/100
    
    plt.imshow(new_image)
    plt.gca().add_patch(plt.Rectangle((x_cd, y_cd), width, height, edgecolor='r', linewidth=2, fill=False))
    plt.savefig('/Users/HP/src/feet_fracture_data/complete')
    plt.show()




def main():
    
    csv_path = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'
    df = pd.read_csv(csv_path)

#    images = df['image'].to_list()
#    x_cds = df['value_x'].to_list()
#    y_cds = df['value_y'].to_list()
#    heights = df['value_width'].to_list()
#    widths = df['value_height'].to_list()


    #image_paths = ['/Users/HP/src/feet_fracture_data/P001 SAGT1/MRI ANKLE (LEFT) W_O CONT_5891215/'+str(image) for image in images]
   
    for index, row in df.iterrows():
        image_path = '/Users/HP/src/feet_fracture_data/P001 SAGT1/MRI ANKLE (LEFT) W_O CONT_5891215/' + row['image']
        x_cd = row['value_x']
        y_cd = row['value_y']
        width = row['value_width']
        height = row['value_height']

        plot_bbox(image_path, x_cd, y_cd, width, height)


if __name__ == "__main__":
    main()