import pandas as pd
import os
import numpy as np

def split_csv_with_sliding_window(input_csv, output_dir, window_size_frames=64, step_size_frames=16):
    df = pd.read_csv(input_csv)
    total_frames = len(df)
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    base_name_split= base_name.split("_")
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    if 'frame' not in df.columns:
        raise ValueError("Input CSV must contain a 'frame' column.")
    for start in range(0, total_frames - window_size_frames + 1, step_size_frames):
        end = start + window_size_frames
        window_df = df.iloc[start:end]
        out_npy = os.path.join(output_dir, f"{str(int(base_name_split[0])+int(df['frame'][start]))}_to_{str(int(base_name_split[0])+int(df['frame'][end-1]))}_{count}_64frames.npy")
        np.save(out_npy, window_df.to_numpy())
        count += 1