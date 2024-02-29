import csv

def count_non_zero_values(csv_file, column_name, limit=100):
    non_zero_count = 0
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            if i >= limit:
                break
            
            value = row.get(column_name)
            if value is not None and value != '' and float(value) != 0:
                non_zero_count += 1
    
    return non_zero_count


csv_path = '/Users/HP/src/feet_fracture_data/json_data/sag_T1_json_columns/annotations/results_box_cleaned_calculation.csv'  
column_name = 'value_rotation' 

count = count_non_zero_values(csv_path, column_name)
print(f"Number of non-zero values in column '{column_name}': {count}")