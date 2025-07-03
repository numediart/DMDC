import pympi
import pandas as pd
from pathlib import Path
import os
from pydub import AudioSegment
import subprocess
import sys

# PATHS
video_dir = Path("V0DataSet/mp4")
csv_dir = Path("test/clean_segments")
tmp_wav_base = Path("V0DataSet/tmp_wav")

# EAF PATH
eaf_dir = Path("Benchmark/ELAN/Dataset_Bench_Manual")
# Directory output clean segments
output_dir = Path("test/clean_segments")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_clean_segments(annotations):
    transcriptions = [ann for ann in annotations if ann['type'] == 'transcription']
    clean_segments = []

    for i, seg in enumerate(transcriptions):
        overlapping = False
        for j, other in enumerate(transcriptions):
            if i == j:
                continue
            if seg['speaker_id'] != other['speaker_id']:
                if not (seg['end_ms'] <= other['start_ms'] or seg['start_ms'] >= other['end_ms']):
                    overlapping = True
                    break
        if not overlapping:
            clean_segments.append(seg)

    return clean_segments

for eaf_file in eaf_dir.glob("*.eaf"):
    eaf_obj = pympi.Elan.Eaf(eaf_file)

    annotations = []
    for tier_name in eaf_obj.get_tier_names():
        if 'SPEAKER_' in tier_name:
            for start, end, text in eaf_obj.get_annotation_data_for_tier(tier_name):
                annotations.append({
                    'type': 'transcription',
                    'speaker_id': tier_name,
                    'start_ms': start,
                    'end_ms': end,
                    'label': text
                })

    clean_segments = extract_clean_segments(annotations)

    df = pd.DataFrame([
        {
            'speaker': seg['speaker_id'],
            'start_ms': seg['start_ms'],
            'end_ms': seg['end_ms'],
            'text': seg['label']
        }
        for seg in clean_segments
    ])

    base_name = eaf_file.stem.replace("_video", "")
    output_csv = output_dir / f"{base_name}_clean_segments.csv"
    df.to_csv(output_csv, index=False)

print("✅ Extraction done, CSV in test/clean_segments/")