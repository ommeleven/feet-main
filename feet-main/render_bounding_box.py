import os
import matplotlib.pyplot as plt
import pandas as pd

from path import my_path


def plot_bbox(image_path, x_cd, y_cd, width, height, angle):
    new_image = plt.imread(image_path)

    # recalculation of all values by percent 
    original_width_img1 = 640
    original_height_img1 = 640
    
    original_width_img2to4 = 864
    original_height_img2to4 = 864

    original_width_img5 = 768
    original_height_img5 = 768

    x_cd = x_cd * original_width_img2to4/100
    y_cd = y_cd * original_height_img2to4/100
    width = width * original_width_img2to4/100
    height = height * original_height_img2to4/100
    angle = angle
    
    plt.imshow(new_image)
    
    plt.gca().add_patch(plt.Rectangle((x_cd, y_cd), width, height, angle=angle, edgecolor='r', linewidth=1, fill=False))

    # also save the image in a folder
    plt.show()




def main():
    
    csv_path = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        image = df['image'].to_string(index=False)
        image_path = os.path.abspath(image)
        print('image_path: ', image_path)
        break
'''
#    images = df['image'].to_list()
#    x_cds = df['value_x'].to_list()
#    y_cds = df['value_y'].to_list()
#    heights = df['value_width'].to_list()
#    widths = df['value_height'].to_list()

    #image_paths = ['/Users/HP/src/feet_fracture_data/P001 SAGT1/MRI ANKLE (LEFT) W_O CONT_5891215/'+str(image) for image in images]
    #subset_df = df.iloc[78:79]
    
    for index, row in subset_df.iterrows():
        
        image_path = '/Users/HP/src/feet_fracture_data/P093 SAGT1/MRI ANKLE (LEFT) W_O CONT_5227736/' + row['image']
        x_cd = row['value_x']
        y_cd = row['value_y']
        width = row['value_width']
        height = row['value_height']
        angle = row['value_rotation']
        #angle = 25
    
    image_path_1 = '/Users/HP/src/feet_fracture_data/P093 SAGT1/MRI ANKLE (LEFT) W_O CONT_5227736/P093 SAGT1_012.jpg'
    image_path_2 = '/Users/HP/src/feet_fracture_data/P056 SAGT1/MRI ANKLE (LEFT) W_O CONT_5660928/P056 SAGT1_007.jpg'
    image_path_3 = '/Users/HP/src/feet_fracture_data/P071 SAGT1/MRI ANKLE (RIGHT) W_O CONT_5499874/P071 SAGT1_010.jpg'
    image_path_4 = '/Users/HP/src/feet_fracture_data/P081 SAGT1/MRI ANKLE (LEFT) W_O CONT_5390016/P081 SAGT1_014.jpg'
    image_path_5 = '/Users/HP/src/feet_fracture_data/P094 SAGT1/MRI ANKLE (LEFT) W_O CONT_5161319/P094 SAGT1_015.jpg'
    image_path_6 = '/Users/HP/src/feet_fracture_data/P028 SAGT1/MRI ANKLE (LEFT) W_O CONT_5814932/P028 SAGT1_015.jpg'

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
    #angle2 = 0

    x_cd3 = 85.12949601
    y_cd3 = 71.86164346
    width3 = 2.287633373
    height3 =  10.55276382
    angle3 = 35.1455455	

    x_cd4 = 27.30806449
    y_cd4 = 61.54141892
    width4 = 2.344150774
    height4 = 8.358208955
    angle4 = 332.6125778	
    
    x_cd5 = 28.49476121
    y_cd5 = 60.23490845
    width5 = 2.40609887
    height5 = 6.088280061
    angle5 = 329.2158535
    # image 5 has dimensions 768 x 768
    x_cd6 = 89.16614457
    y_cd6 = 79.83503301
    width6 = 2.550695498
    height6 = 8.797701843
    angle6 = 22.83365418
    							
    plot_bbox(image_path_6, x_cd6, y_cd6, width6, height6, angle6)
    '''
if __name__ == "__main__":
    save_path = my_path
    main()