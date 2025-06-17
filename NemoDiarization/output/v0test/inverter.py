import os
import pandas as pd

# Folder containing the CSV files
folder_path = '/home/hugo-mny/UMONS/DMDC/NemoDiarization/output/v0test/'

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Swap 'speaker_00' with 'speaker_01'
        df.replace({'SPEAKER_00': 'SPEAKER_01', 'SPEAKER_01': 'SPEAKER_00'}, inplace=True)
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

print("Speaker swapping completed for all CSV files.")