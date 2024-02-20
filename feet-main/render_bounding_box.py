import os
import matplotlib.pyplot as plt
import pandas as pd

from path import my_path


def plot_bbox(image_path, x_cd, y_cd, width, height, angle):
    new_image = plt.imread(image_path)

    # recalculation of all values by percent 
    original_width_img1 = 640
    original_height_img1 = 640
    
    original_width_img2 = 864
    original_height_img2 = 864

    x_cd = x_cd * original_width_img2/100
    y_cd = y_cd * original_height_img2/100
    width = width * original_width_img2/100
    height = height * original_height_img2/100
    angle = angle
    
    plt.imshow(new_image)
    
    plt.gca().add_patch(plt.Rectangle((x_cd, y_cd), width, height, angle=angle, edgecolor='r', linewidth=2, fill=False))
    #save_path = os.path.join('/Users/HP/src/feet_fracture_data/complete/', "bbox.png") 
    #plt.savefig('/Users/HP/src/feet_fracture_data/complete/')

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
    #subset_df = df.iloc[78:79]
    '''
    for index, row in subset_df.iterrows():
        
        image_path = '/Users/HP/src/feet_fracture_data/P093 SAGT1/MRI ANKLE (LEFT) W_O CONT_5227736/' + row['image']
        x_cd = row['value_x']
        y_cd = row['value_y']
        width = row['value_width']
        height = row['value_height']
        angle = row['value_rotation']
        #angle = 25
    '''
    image_path_1 = '/Users/HP/src/feet_fracture_data/P093 SAGT1/MRI ANKLE (LEFT) W_O CONT_5227736/P093 SAGT1_012.jpg'
    image_path_2 = '/Users/HP/src/feet_fracture_data/P056 SAGT1/MRI ANKLE (LEFT) W_O CONT_5660928/P056 SAGT1_007.jpg'
    x_cd1 = 85.07191759
    y_cd1 = 73.01503272
    width1 = 2.190576548
    height1 = 9.150314143
    angle1 = 27.02463313

    x_cd2 = 27.84171809
    y_cd2 = 76.47347111
    width2 = 2.283105023
    height2 = 3.348554033
    angle2 = 333.8531588
    								
    plot_bbox(image_path_2, x_cd2, y_cd2, width2, height2, angle2)


if __name__ == "__main__":
    save_path = my_path
    main()