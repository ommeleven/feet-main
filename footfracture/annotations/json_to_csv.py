import os
from footfracture.config import DATA_DIR
import pandas as pd
import ast

df = pd.read_csv(os.path.join(DATA_DIR, "json_to_ex.csv"))
res_df = pd.DataFrame(index=df.index)
res_df['image'] = df['data.image']
for index,row in df.iterrows():
    res_df.at[index, 'image'] = res_df.at[index, 'image'][24:]

#print(res_df.head())

res_df['label'] = df['annotations.result']

for index,row in df.iterrows():
   #if df.at[index, 'label']!='Healthy':
    #print(type(res_df.at[index, 'label']))
    #print(res_df.at[index, 'label'])
    data = ast.literal_eval(res_df.at[index, 'label'])
    if data == []:
        res_df.at[index, 'label'] = 'Healthy'
    else:
        res_df.at[index, 'label'] = data[0]['value']['rectanglelabels'][0]
        res_df.at[index, 'x'] =  data[0]['value']['x']
        res_df.at[index, 'y'] =  data[0]['value']['y']
        res_df.at[index, 'width'] = data[0]['value']['width']
        res_df.at[index, 'height'] = data[0]['value']['height']
        res_df.at[index, 'original_width'] = data[0]['original_width']
        res_df.at[index, 'original_height'] =  data[0]['original_height']
    
res_df.to_csv(os.path.join(DATA_DIR, "output.csv"))
print(res_df) 
#img_names = df.iloc[df['data.image']]
#for img_name in img_names:
#    print('names:', img_name)
#labels
#coords
#rotation

