import numpy as np
import glob
import os



def merge_numpy_files(output_filename="V0.5_DMDC_11K_Segment_listener.npy"):
    # Directory containing the .npy files
    data_dir = os.path.dirname(__file__)

    # Find all .npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*listener.npy")))

    # Load all numpy arrays and collect them in a list
    arrays = [np.load(f) for f in npy_files]

    # Concatenate along the first axis
    merged = np.concatenate(arrays, axis=0)

    # Print the shape and a preview
    print("Shape:", merged.shape)
    print(merged)

    # Optionally, save the merged array
    np.save(os.path.join(data_dir, output_filename), merged)