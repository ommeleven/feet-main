import numpy as np
import os

def calculate_metrics(confusion_matrix):
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)
    
    # Sensitivity, Specificity, Accuracy, Loss
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    total_accuracy = (TP + TN) / np.sum(confusion_matrix)
    total_accuracy = TP / np.sum(confusion_matrix)
    
    #total_loss = (FP + FN) / np.sum(confusion_matrix)
    
    metrics = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Total Accuracy': total_accuracy,
        
    }
    
    # Create the metrics string in a table format
    metrics_str = "Metric          Class 1         Class 2         Class 3\n"
    for key, value in metrics.items():
        metrics_str += f"{key:<15}"
        for v in value:
            metrics_str += f"{v:<15.6f}"
        metrics_str += "\n"
    
    return metrics_str

# Example usage:


conf_matrix = np.array([[14, 33 , 2],
 [53 ,29, 19],
 [ 6  ,6 ,54]])
metrics_output = calculate_metrics(conf_matrix)
print(metrics_output)
1

def count_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return 0

    # Initialize a counter
    file_count = 0
    
    # Iterate over all items in the folder
    for item in os.listdir(folder_path):
        # Construct full item path
        item_path = os.path.join(folder_path, item)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(item_path):
            file_count += 1
    
    return file_count

folder_path = '/home/odange/repo/feet_fracture_data/feet_fracture_data/sequential/healthy-unhealthy/train/healthy'
#file_count = count_files_in_folder(folder_path)
#print(f"Total number of files in '{folder_path}': {file_count}")
