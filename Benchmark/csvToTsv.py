import pandas as pd
import os
import glob

# Get all CSV files in the V0DataSet/segments/ folder
csv_files = glob.glob("V0DataSet/segments/*.csv")

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file, quotechar='"')
    
    # Create output filename by replacing .csv with .tsv
    output_file = os.path.basename(csv_file).replace('.csv', '.tsv')
    
    # Save as TSV
    df.to_csv(output_file, sep="\t", index=False)
    
    print(f"Converted {csv_file} to {output_file}")