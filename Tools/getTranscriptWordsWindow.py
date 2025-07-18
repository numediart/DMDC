import os
import numpy as np
import pandas as pd
import re

def find_matching_csv(transcript_folder, start_ms, end_ms):
    """
    Searches for the CSV file in transcript_folder whose window overlaps [start_ms, end_ms].
    """
    csv_files = [f for f in os.listdir(transcript_folder) if f.endswith("_words.csv")]
    for csv_file in csv_files:
        # Extracts timestamps from the filename, e.g.: segment_0001_35030ms_41830ms_words.csv
        m = re.match(r"segment_\d+_(\d+)ms_(\d+)ms_words\.csv", csv_file)
        if not m:
            continue
        csv_start_ms = int(m.group(1))
        csv_end_ms = int(m.group(2))
        # Checks for window overlap
        if not (end_ms < csv_start_ms or start_ms > csv_end_ms):
            return os.path.join(transcript_folder, csv_file), csv_start_ms, csv_end_ms
    return None, None, None

def get_words_in_windows_all_segments(transcript_folder, windows_folder, fps, output_txt_folder=None):
    if output_txt_folder is None:
        output_txt_folder = windows_folder.replace("n_frames_windowed_clips", "windowed_transcripts")
    os.makedirs(output_txt_folder, exist_ok=True)

    for filename in os.listdir(windows_folder):
        if not filename.endswith(".npy"):
            continue

        parts = filename.split("_")
        try:
            start_frame = int(parts[0])
            end_frame = int(parts[2])
        except Exception:
            print(f"[WARN] Malformed filename: {filename}")
            continue

        start_ms = start_frame * 1000 / fps
        end_ms = end_frame * 1000 / fps

        # Find the CSV file that overlaps with the window [start_ms, end_ms]
        matching_csv, csv_start_ms, csv_end_ms = find_matching_csv(transcript_folder, start_ms, end_ms)
        if matching_csv is None:
            print(f"[WARN] No CSV found overlapping window {start_ms}-{end_ms} ms")
            continue

        df = pd.read_csv(matching_csv)
        if not {'start', 'end', 'word'}.issubset(df.columns):
            print(f"[ERR] Malformed CSV: {matching_csv}")
            continue

        # Calculate absolute word timings
        df['abs_start'] = csv_start_ms + df['start'] * 1000
        df['abs_end'] = csv_start_ms + df['end'] * 1000

        # Keep words that touch the window [start_ms, end_ms]
        overlap = df[(df['abs_end'] >= start_ms) & (df['abs_start'] <= end_ms)]
        words = overlap['word'].tolist()

        # Save
        txt_filename = filename.replace(".npy", ".txt")
        txt_path = os.path.join(output_txt_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(words))

