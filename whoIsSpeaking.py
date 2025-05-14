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
        print("GPU available : ",torch.cuda.get_device_name(0))  
    else:
        print("GPU not available. Running on CPU.")



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
    parent_folder = os.path.basename(os.path.dirname(filename))      # 1_video
    output_dir = os.path.join("V0DataSet", "Diarization_Results", parent_folder)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pathname = os.path.join(output_dir, f"{base_filename}_diarization_results_{timestamp}.csv")
    df.to_csv(pathname, index=False)
    print(f"Saved diarization results here: {pathname}")

    # Convert CSV as subtitle file and save it
    subtitle_output = os.path.join("V0DataSet", "Subtitle", parent_folder)
    os.makedirs(subtitle_output, exist_ok=True)
    csv_to_subtitle(pathname, os.path.join(subtitle_output, f"{base_filename}.srt"))

    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")
