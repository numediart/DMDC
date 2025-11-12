import numpy as np
import glob
from config import BASE_DIR
import os

def merge_numpy_files(input_directory, output_path):
    # Directory containing the .npy files
    data_dir = input_directory

    # Find all .npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

    # Load all numpy arrays and collect them in a list
    arrays = [np.load(f) for f in npy_files]

    # Concatenate along the first axis
    merged = np.concatenate(arrays, axis=0)

    # Print the shape and a preview
    print("Shape:", merged.shape)
    print(merged)

    # Optionally, save the merged array
    np.save(output_path, merged)


if __name__ == "__main__":
    # Example usage
    dataset="V0.10DataSet"
    for i in range(1, 11):
        input_directory = os.path.join(BASE_DIR, dataset, "mfcc_output", f"{i}_video")
        output_path = os.path.join(BASE_DIR, dataset, f"{i}_video_mfcc.npy")
        merge_numpy_files(input_directory, output_path)
        input_directory = os.path.join(BASE_DIR, dataset, "n_frames_windowed_clips", f"{i}_video", "listener")
        output_path = os.path.join(BASE_DIR, dataset, f"{i}_video_AU_listener.npy")
        merge_numpy_files(input_directory, output_path)
        input_directory = os.path.join(BASE_DIR, dataset, "n_frames_windowed_clips", f"{i}_video", "speaker")
        output_path = os.path.join(BASE_DIR, dataset, f"{i}_video_AU_speaker.npy")
        merge_numpy_files(input_directory, output_path)

