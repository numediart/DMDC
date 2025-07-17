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
            return os.path.join(transcript_folder, csv_file)
    return None

def get_words_in_windows_all_segments(transcript_folder, windows_folder, fps=25, output_txt_folder=None):
    """
    For each .npy window file (e.g.: 26949_to_27076_31_64frames.npy),
    retrieves the words in the transcription CSV files that overlap with this window.
    Saves the transcription in a .txt file next to the .npy.
    """

    if output_txt_folder is None:
        output_txt_folder = windows_folder.replace("n_frames_windowed_clips", "windowed_transcripts")

    os.makedirs(output_txt_folder, exist_ok=True)

    # Search for all .npy files in the speaker/ folder
    for filename in os.listdir(windows_folder):
        if not filename.endswith(".npy"):
            continue

        parts = filename.split("_")
        try:
            start_ms = int(parts[0])
            end_ms = int(parts[2])
        except Exception:
            print(f"[WARN] Malformed filename: {filename}")
            continue

        start_time = start_ms / 1000.0
        end_time = end_ms / 1000.0

        matching_csv = find_matching_csv(transcript_folder, start_ms, end_ms)
        if matching_csv is None:
            print(f"[WARN] No CSV found overlapping window {start_ms}-{end_ms} ms in {transcript_folder}")
            continue

        try:
            df = pd.read_csv(matching_csv)
        except Exception as e:
            print(f"[ERR] Problem reading CSV {matching_csv}: {e}")
            continue

        if not {'start', 'end', 'word'}.issubset(df.columns):
            print(f"[ERR] Malformed CSV: {matching_csv}")
            continue

        words = df[(df['end'] >= start_time) & (df['start'] <= end_time)]['word'].tolist()

        txt_filename = filename.replace(".npy", ".txt")
        txt_path = os.path.join(output_txt_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(words))
