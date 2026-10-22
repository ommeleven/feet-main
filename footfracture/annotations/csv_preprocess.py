import os
from footfracture.config import DATA_DIR
import pandas as pd

# Read the CSV file
df = pd.read_csv(os.path.join(DATA_DIR, "output.csv"))

# Replace missing or empty 'Label' values (including NaN and whitespace) with "Healthy"
df['Label'] = df['Label'].replace(to_replace=['', ' ', None], value='Healthy').fillna('Healthy')

# Print the images where the label was updated
for image in df.loc[df['Label'] == 'Healthy', 'Image']:
    print(image)

# Save the updated DataFrame to a new file
df.to_csv(os.path.join(DATA_DIR, "output_updated.csv"), index=False)
