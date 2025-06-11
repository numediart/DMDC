import os
import numpy as np
import pandas as pd
import re

def csvs_to_npy(csv_folder, output_name, max_size=None):
    """
    Reads all CSV files in the given folder, stacks them along a new axis (3D array),
    and saves the merged data as a .npy file in the current directory.
    If max_size is defined, only the first max_size columns are kept from each CSV (not rows).
    """
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    def natural_sort_key(s):
        # Split string into list of strings and integers for natural sorting
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    csv_files.sort(key=natural_sort_key)

    # Read all CSV files and collect as numpy arrays
    arrays = []
    max_rows = 0
    max_cols = 0
    # First pass: determine max shape
    for file in csv_files:
        df = pd.read_csv(os.path.join(csv_folder, file))
        if max_size is not None:
            df = df.iloc[:max_size, :]
        max_rows = max(max_rows, df.shape[0])
        max_cols = max(max_cols, df.shape[1])
        arrays.append(df.values)
    # Second pass: pad arrays to max shape
    padded_arrays = []
    for arr in arrays:
        pad_rows = max_rows - arr.shape[0]
        pad_cols = max_cols - arr.shape[1]
        padded = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
        padded_arrays.append(padded)
    # Stack arrays along a new axis (axis=0)
    merged_array = np.stack(padded_arrays, axis=0)

    # Save as .npy file
    np.save(output_name, merged_array)

if __name__ == "__main__":
    for i in range(1,2):
        print(i)
        # csvs_to_npy("./mfcc_output/"+str(i)+"_video")
        csvs_to_npy("./64_frames_windowed_clips/"+str(i)+"_video/speaker",str(i)+"_video_AU_speaker.npy")
        csvs_to_npy("./64_frames_windowed_clips/"+str(i)+"_video/listener",str(i)+"_video_AU_listener.npy")
        # csvs_to_npy("./mfcc_output/"+str(i)+"_video/",str(i)+"_video_MFCC.npy",128)
