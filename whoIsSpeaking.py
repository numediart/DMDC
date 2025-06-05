from pyannote.audio import Pipeline
import pandas as pd
import os
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
from csvToSubTiltle import csv_to_subtitle
import time
#Constants
DATASET_FOLDER="V0.5DataSet"

def run_diarization(filename,datasetName=DATASET_FOLDER):
    start_time = time.time()
    ##################################################################
    #Credit https://github.com/pyannote/pyannote-audio README.md
    ##################################################################
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")

    # send pipeline to GPU (when available)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("[Diarization/pyTorch] GPU available : ",torch.cuda.get_device_name(0))  
    else:
        print("[Diarization/pyTorch] GPU not available. Running on CPU.")



    # apply pretrained pipeline with 2 speakers
    with ProgressHook() as hook:
        diarization = pipeline(filename,min_speakers=0,hook=hook)



    # Create a dataframe from the diarization results
    data = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        data.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    df = pd.DataFrame(data)

    # Display the dataframe
    print(df)



    # Save dataframe as CSV with file name and timestamp
    base_filename = os.path.splitext(os.path.basename(filename))[0]  # clip_001
    output_dir = os.path.join(datasetName, "Diarization_Results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pathname = os.path.join(output_dir, f"{base_filename}_diarization_results_{timestamp}.csv")
    df.to_csv(pathname, index=False)
    print(f"[Diarization] Saved diarization results here: {pathname}")

    # Convert CSV as subtitle file and save it
    subtitle_output = os.path.join(datasetName, "Subtitle")
    os.makedirs(subtitle_output, exist_ok=True)
    csv_to_subtitle(pathname, os.path.join(subtitle_output, f"{base_filename}.srt"))

    end_time = time.time()
    print("[Diarization] Execution time:", end_time - start_time, "seconds")


def assign_speakers_to_segments_from_df(visual_segments, diarization_df):
    result_segments = []

    for v_start, v_end, category in visual_segments:
        overlap = diarization_df[
            (diarization_df['end'] > v_start) & (diarization_df['start'] < v_end)
        ]

        if overlap.empty:
            result_segments.append((v_start, v_end, category, "NA"))
        else:
            for _, row in overlap.iterrows():
                seg_start = max(v_start, row['start'])
                seg_end = min(v_end, row['end'])
                speaker = row['speaker']
                result_segments.append((seg_start, seg_end, category, speaker))

    return result_segments

def merge_contiguous_segments(segments, max_gap):
    if not segments:
        return []

    merged = [segments[0]]

    for current in segments[1:]:
        last = merged[-1]
        last_end = float(last[1])
        current_start = float(current[0])

        if last[2] == current[2] and last[3] == current[3] and (current_start - last_end) <= max_gap:
            merged[-1] = (last[0], current[1], last[2], last[3])
        else:
            merged.append(current)

    return merged

def filter_short_segments(segments, min_duration=0.5):
    return [seg for seg in segments if float(seg[1]) - float(seg[0]) >= min_duration]