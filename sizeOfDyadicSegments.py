import os
import pandas as pd



def calculate_dyadic_length(input_folder):
    total_length = 0

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            
            df = pd.read_csv(file_path)
            
            # Filter rows where 'category' is 'dyadic' and calculate the total length
            dyadic_rows = df[df['category'] == 'dyadic']
            total_length += (dyadic_rows['end_time'] - dyadic_rows['start_time']).sum()
    
    return total_length

# Example usage
input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./V0.2DataSet/segments/")
dyadic_length = calculate_dyadic_length(input_folder)
print(f"Total length of dyadic segments: {dyadic_length}")
