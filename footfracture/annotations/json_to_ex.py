import os
from footfracture.config import DATA_DIR
import pandas as pd

# Input and output file paths
input_file = os.path.join(DATA_DIR, "createml.csv")  # Replace with your input file name
output_file = "cleaned_annotations.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Prepare a dictionary to hold combined data
combined_data = {}

# Process each row in the DataFrame
for _, row in df.iterrows():
    image = row["image"]
    annotations = []
    for i in range(6):  # Assuming a maximum of 6 annotations per image
        label_col = f"annotations/{i}/label"
        if pd.notna(row[label_col]):  # Check if the label exists
            annotation = {
                "Label": row[label_col],
                "Height": row[f"annotations/{i}/coordinates/height"],
                "Width": row[f"annotations/{i}/coordinates/width"],
                "X": row[f"annotations/{i}/coordinates/x"],
                "Y": row[f"annotations/{i}/coordinates/y"]
            }
            annotations.append(annotation)

    # If no annotations were found, assign the "healthy" label
    if not annotations:
        annotations.append({
            "Label": "healthy",
            "Height": "",
            "Width": "",
            "X": "",
            "Y": ""
        })

    if image not in combined_data:
        combined_data[image] = []
    combined_data[image].extend(annotations)

# Prepare data for the new DataFrame
output_rows = []
for image, annotations in combined_data.items():
    combined_labels = "; ".join([f"{ann['Label']} (H:{ann['Height']}, W:{ann['Width']}, X:{ann['X']}, Y:{ann['Y']})" for ann in annotations])
    output_rows.append({"Image": image, "Annotations": combined_labels})

# Create a new DataFrame with the combined data
output_df = pd.DataFrame(output_rows)

# Save the combined data to a new CSV file
output_df.to_csv(output_file, index=False)

print(f"Combined annotations have been saved to {output_file}")
