import os
import pandas as pd
from moviepy import VideoFileClip
from pathlib import Path
import subprocess
import sys

# PATH
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "V0DataSet" / "mp4"
SEGMENTS_DIR = BASE_DIR / "test" / "clean_segments"
TMP_WAV_DIR = BASE_DIR / "test" / "tmp_wav"

TMP_WAV_DIR.mkdir(parents=True, exist_ok=True)

def extract_wav_segments(video_idx):
    video_path = VIDEO_DIR / f"{video_idx}_video.mp4"
    segment_csv = SEGMENTS_DIR / f"{video_idx}_clean_segments.csv"
    output_dir = TMP_WAV_DIR / str(video_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(segment_csv)

    video = VideoFileClip(str(video_path))

    segment_paths = []

    for i, row in df.iterrows():
        start_time = float(row["start_ms"])
        stop_time = float(row["end_ms"])
        # Convert from ms to seconds
        start_time /= 1000.0
        stop_time /= 1000.0
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

        # Transcript
        subprocess.run([
            sys.executable,
            "transcribe_parakeet.py",
            video_idx,
            *segment_paths
        ])

if __name__ == "__main__":
    pd.read_csv("")