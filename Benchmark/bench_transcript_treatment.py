import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip

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
    segment_paths = []

    for i, row in df.iterrows():
        start_time = float(row["start_ms"]) / 1000.0
        stop_time = float(row["end_ms"]) / 1000.0
        segment_clip = video.subclipped(start_time, stop_time)

        segment_path = output_dir / f"seg_{i:04d}.wav"
        segment_clip.audio.write_audiofile(str(segment_path), codec='pcm_s16le', logger=None)
        segment_paths.append(str(segment_path))

    video.close()
    return segment_paths

def process_all_segments():
    for csv_file in SEGMENTS_DIR.glob("*_clean_segments.csv"):
        video_idx = csv_file.stem.split("_")[0]
        print(f"[INFO] Processing video {video_idx}")

        segment_paths = extract_wav_segments(video_idx)

        if not segment_paths:
            print(f"[WARNING] No segments extracted for video {video_idx}")
            continue

        # Call transcription script
        subprocess.run([
            sys.executable,
            str(BASE_DIR.parent / "Transcribe" / "transcribe_distillarge.py"),
            video_idx,
            *segment_paths
        ])

if __name__ == "__main__":
    process_all_segments()
