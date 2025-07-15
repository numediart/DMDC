import os
import numpy as np


def print_shapes_and_preview(folder_path, num_lines=5):
    """
    Prints the shapes and first few lines of all .npy files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        num_lines (int): Number of lines to preview from each .npy file.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through files
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            try:
                # Load the .npy file
                data = np.load(file_path, allow_pickle=True)
                # Print the shape of the array
                print(f"{file}: {data.shape}")
                # Print the first few lines of the array
                print(f"Preview (first {num_lines} lines):")
                print(data[:num_lines])
            except Exception as e:
                print(f"Error loading {file}: {e}")

# Specify the folder containing .npy files
folder_path = "DataSetCleaningTools/cleaned/"
print_shapes_and_preview(folder_path)