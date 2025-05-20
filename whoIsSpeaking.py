from pyannote.audio import Pipeline
import pandas as pd
import os
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
from csvToSubTiltle import csv_to_subtitle
import time
#Constants

def run_diarization(filename):
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
        diarization = pipeline(filename,max_speakers=2,min_speakers=0,hook=hook)



    # Create a dataframe from the diarization results
    data = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        data.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    df = pd.DataFrame(data)

    # Display the dataframe
    print(df)



    # Save dataframe as CSV with file name and timestamp
    base_filename = os.path.splitext(os.path.basename(filename))[0]  # clip_001
    output_dir = os.path.join("V0DataSet", "Diarization_Results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pathname = os.path.join(output_dir, f"{base_filename}_diarization_results_{timestamp}.csv")
    df.to_csv(pathname, index=False)
    print(f"Saved diarization results here: {pathname}")

    # Convert CSV as subtitle file and save it
    subtitle_output = os.path.join("V0DataSet", "Subtitle")
    os.makedirs(subtitle_output, exist_ok=True)
    csv_to_subtitle(pathname, os.path.join(subtitle_output, f"{base_filename}.srt"))

    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")


def assign_speakers_to_segments_from_df(visual_segments, diarization_df):
    result_segments = []

    for v_start, v_end, category in visual_segments:
        overlap = diarization_df[
            (diarization_df['end'] > v_start) & (diarization_df['start'] < v_end)
        ]

        if overlap.empty:
            speaker = "NA"
        else:
            # Calculate the speaking duration per speaker
            speaker_durations = {}
            for _, row in overlap.iterrows():
                overlap_start = max(v_start, row['start'])
                overlap_end = min(v_end, row['end'])
                duration = overlap_end - overlap_start

                speaker_durations[row['speaker']] = speaker_durations.get(row['speaker'], 0) + duration

            if len(speaker_durations) == 1:
                speaker = list(speaker_durations.keys())[0]
            else:
                # If there is a tie or multiple speakers, choose the one with the longest duration (or NA based on logic)
                sorted_durations = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
                if sorted_durations[0][1] - sorted_durations[1][1] < 0.5:
                    speaker = "NA"
                else:
                    speaker = sorted_durations[0][0]

        result_segments.append((v_start, v_end, category, speaker))

    return result_segments
