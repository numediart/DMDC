import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip
import time
import csv

# PATHS
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR.parent / "V0DataSet" / "mp4"
SEGMENTS_DIR = BASE_DIR.parent / "test" / "clean_segments"
TMP_WAV_DIR = BASE_DIR.parent / "test" / "tmp_wav"

TMP_WAV_DIR.mkdir(parents=True, exist_ok=True)

def extract_wav_segments(video_idx):
    video_path = VIDEO_DIR / f"{video_idx}_video.mp4"
    segment_csv = SEGMENTS_DIR / f"{video_idx}_clean_segments.csv"
    output_dir = TMP_WAV_DIR / str(video_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not segment_csv.exists():
        print(f"[WARNING] Segment CSV not found: {segment_csv}")
        return []

    df = pd.read_csv(segment_csv)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return []

    video = VideoFileClip(str(video_path))
    duration = video.duration
    segment_paths = []

    for i, row in df.iterrows():
        start_time = float(row["start_ms"]) / 1000.0
        stop_time = float(row["end_ms"]) / 1000.0

        if stop_time > duration:
            print(f"[WARNING] Segment {i} stop_time ({stop_time:.2f}s) exceeds video duration ({duration:.2f}s), truncating.")
            stop_time = duration - 0.01  # garde une petite marge de sécurité

        if start_time >= stop_time:
            print(f"[ERROR] Invalid segment {i}: start_time ({start_time:.2f}) >= stop_time ({stop_time:.2f}), skipping.")
            continue

        try:
            segment_clip = video.subclipped(start_time, stop_time)
            segment_path = output_dir / f"seg_{i:04d}.wav"
            segment_clip.audio.write_audiofile(str(segment_path), codec='pcm_s16le', logger=None)
            segment_paths.append(str(segment_path))
        except Exception as e:
            print(f"[ERROR] Failed to process segment {i}: {e}")

    video.close()
    return segment_paths

def process_all_segments():
    exec_time_csv = BASE_DIR / "execution_times.csv"
    if not exec_time_csv.exists():
        with open(exec_time_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_idx", "execution_time_seconds"])
    for csv_file in SEGMENTS_DIR.glob("*_clean_segments.csv"):
        start = time.time()
        video_idx = csv_file.stem.split("_")[0]
        print(f"[INFO] Processing video {video_idx}")

        segment_paths = extract_wav_segments(video_idx)

        if not segment_paths:
            print(f"[WARNING] No segments extracted for video {video_idx}")
            continue

        # Call transcription script
        subprocess.run([
            sys.executable,
            str(BASE_DIR.parent / "Transcribe" / "transcribe_parakeet.py"),
            video_idx,
            *segment_paths
        ])
        end = time.time()
        elapsed = end - start
        print(f"Execution time : {end - start:.2f} seconds")
        # Write execution time to CSV
        with open(exec_time_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([video_idx, f"{elapsed:.2f}"])

if __name__ == "__main__":
    process_all_segments()
